import cv2
import numpy as np
import pytesseract
import argparse
from imutils.object_detection import non_max_suppression

def load_yolo():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, output_layers

def load_east():
    net = cv2.dnn.readNet("frozen_east_text_detection.pb")
    return net

def detect_people(frame, net, output_layers, classes):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # Only consider the 'person' class
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    person_count = len(indexes)
    return frame, person_count

def preprocess_for_ocr(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    gray = cv2.medianBlur(gray, 3)
    return gray

def detect_text(frame, net):
    orig = frame.copy()
    (H, W) = frame.shape[:2]

    (newW, newH) = (320, 320)
    rW = W / float(newW)
    rH = H / float(newH)

    frame = cv2.resize(frame, (newW, newH))
    blob = cv2.dnn.blobFromImage(frame, 1.0, (newW, newH), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])

    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(numCols):
            if scoresData[x] < 0.5:
                continue

            offsetX = x * 4.0
            offsetY = y * 4.0

            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    boxes = non_max_suppression(np.array(rects), probs=confidences)

    print(f"Detected {len(boxes)} text regions")

    for (startX, startY, endX, endY) in boxes:
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)

        roi = orig[startY:endY, startX:endX]
        roi = preprocess_for_ocr(roi)
        config = ("-l eng --oem 1 --psm 7")
        text = pytesseract.image_to_string(roi, config=config).strip()

        if text:
            print(f"Detected text: {text}")
            cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 0, 255), 2)
            text_y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(orig, text, (startX, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    return orig

def detect(frame, yolo_net, yolo_output_layers, east_net, classes):
    frame, person_count = detect_people(frame, yolo_net, yolo_output_layers, classes)
    frame = detect_text(frame, east_net)
    cv2.putText(frame, f'Total Persons : {person_count}', (40, 70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)
    return frame

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', type=str, help='Path to input image')
    parser.add_argument('-v', '--video', type=str, help='Path to input video file')
    parser.add_argument('-c', '--camera', action='store_true', help='Use camera for live detection')
    args = parser.parse_args()

    classes = ["person"]

    yolo_net, yolo_output_layers = load_yolo()
    east_net = load_east()

    if args.image:
        image = cv2.imread(args.image)
        if image is None:
            print("[ERROR] Image not found or unable to read.")
            return
        frame = detect(image, yolo_net, yolo_output_layers, east_net, classes)
        cv2.imshow('Detection', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print("[ERROR] Unable to open video file.")
            return
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = detect(frame, yolo_net, yolo_output_layers, east_net, classes)
            cv2.imshow('Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    elif args.camera:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Unable to access the camera.")
            return
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to grab frame.")
                break
            frame = detect(frame, yolo_net, yolo_output_layers, east_net, classes)
            cv2.imshow('Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("[INFO] No input source provided. Use -i for image, -v for video, or -c for camera.")

if __name__ == "__main__":
    main()
