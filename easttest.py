import cv2
import numpy as np
import pytesseract
import argparse
from imutils.object_detection import non_max_suppression

def load_east():
    net = cv2.dnn.readNet("frozen_east_text_detection.pb")
    return net

def preprocess_for_ocr(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
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
            cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 0, 255), 2)
            text_y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(orig, text, (startX, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    return orig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', required=True, help='Path to the input image')
    args = parser.parse_args()

    east_net = load_east()

    image = cv2.imread(args.image)
    if image is None:
        print("[ERROR] Image not found or unable to read.")
        return

    frame = detect_text(image, east_net)
    cv2.imshow('Text Detection', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
