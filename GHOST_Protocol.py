import streamlit as st
import cv2
import numpy as np
import pytesseract
from imutils.object_detection import non_max_suppression
from PIL import Image, UnidentifiedImageError
import requests
from io import BytesIO
import tempfile
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import face_recognition

# Load YOLO model
@st.cache_resource
def load_yolo():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, output_layers

# Load EAST text detector
@st.cache_resource
def load_east():
    net = cv2.dnn.readNet("frozen_east_text_detection.pb")
    return net

# Detect people in frame
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

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)  # Use the exact same NMS threshold

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

# Preprocess for OCR
def preprocess_for_ocr(roi):
    if roi is None or roi.size == 0:
        return None
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    gray = cv2.medianBlur(gray, 3)
    return gray

# Detect text in frame
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
        if roi is None:
            continue
        config = ("-l eng --oem 1 --psm 7")
        text = pytesseract.image_to_string(roi, config=config).strip()

        if text:
            cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 0, 255), 2)
            text_y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(orig, text, (startX, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    return orig

# Detection pipeline
def detect(frame, yolo_net, yolo_output_layers, east_net, classes, send_email=False, known_face_encodings=None, known_face_names=None, tolerance=0.6):
    if frame is None or frame.size == 0:
        return None, 0
    frame, person_count = detect_people(frame, yolo_net, yolo_output_layers, classes)
    frame = detect_text(frame, east_net)

    # If known face encodings are provided, check for matches
    if known_face_encodings is not None and known_face_names is not None:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=tolerance)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)

            if send_email and name != "Unknown":
                send_email_notification(person_count, frame, name)

    if send_email and person_count > 0 and (known_face_encodings is None or all(name == "Unknown" for name in known_face_names)):
        send_email_notification(person_count, frame, "Unknown Person")

    return frame, person_count

# Process the image
def process_image(image, send_email=False, known_face_encodings=None, known_face_names=None, tolerance=0.6):
    yolo_net, yolo_output_layers = load_yolo()
    east_net = load_east()
    classes = ["person"]
    return detect(image, yolo_net, yolo_output_layers, east_net, classes, send_email, known_face_encodings, known_face_names, tolerance)

# Load image from URL
def load_image_from_url(url):
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        return img
    except (UnidentifiedImageError, requests.RequestException) as e:
        st.sidebar.error(f"Error loading image: {e}")
        return None

# Send email notification with image
def send_email_notification(person_count, image, detected_name):
    sender_email = "alert.ghost.protocol@gmail.com"
    receiver_email = "divyanshuydv0002@gmail.com"
    subject = "Alert Person Detected"
    body = f"{person_count} person(s) detected, including {detected_name}."

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject

    msg.attach(MIMEText(body, "plain"))

    # Convert image to bytes and attach to email
    _, img_encoded = cv2.imencode('.jpg', image)
    image_bytes = img_encoded.tobytes()
    image_attachment = MIMEImage(image_bytes, name="detected_person.jpg")
    msg.attach(image_attachment)

    try:
        with smtplib.SMTP("smtp.sendgrid.net", 587) as server:
            server.starttls()
            server.login("apikey", "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")
            server.sendmail(sender_email, receiver_email, msg.as_string())
    except Exception as e:
        st.error(f"Error sending email: {e}")

def main():
    st.title("G.H.O.S.T. Protocol: Graphical Human Observation and Surveillance Technology")

    st.sidebar.title("About")
    local_image_path = "/Users/divyanshuyadav/Downloads/DYz.jpg"
    img = Image.open(local_image_path)
    st.sidebar.image(img, width=300)
    st.sidebar.markdown("""
    **Developed by Divyanshu Yadav**  
    An Artificial Intelligence and Data Science Engineering Student with excellent programming skills, 
    and a deep interest in new developments in the field of Artificial Intelligence, Machine Learning, and Data Science.
    """)
    st.sidebar.markdown("[GitHub](https://github.com/DivZyzz)")
    st.sidebar.markdown("[LinkedIn](https://www.linkedin.com/in/divyanshu-yadav-b5427b259/)")
    st.sidebar.markdown("[Portfolio Website](https://divzyzz.github.io/DivyanshuPortfolio/)")

    st.sidebar.title("Options")
    option = st.sidebar.selectbox("Select input source", ("Image", "Video", "Webcam", "Surveillance", "Ghost Protocol"))

    if option == "Image":
        image_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if image_file is not None:
            img = Image.open(image_file)
            img = np.array(img)
            result_img, person_count = process_image(img, send_email=False)
            if result_img is not None:
                # Display person count in the corner
                cv2.putText(result_img, f'People Count: {person_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                st.image(result_img, caption="Processed Image", use_column_width=True)
            else:
                st.error("Error processing the image.")

    elif option == "Video":
        video_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
        if video_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())
            cap = cv2.VideoCapture(tfile.name)

            stframe = st.empty()
            yolo_net, yolo_output_layers = load_yolo()
            east_net = load_east()
            classes = ["person"]

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame, person_count = detect(frame, yolo_net, yolo_output_layers, east_net, classes, send_email=False)
                if frame is not None:
                    # Display person count in the corner
                    cv2.putText(frame, f'People Count: {person_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    stframe.image(frame, channels="BGR")
            cap.release()

    elif option == "Webcam":
        stframe = st.empty()
        cap = cv2.VideoCapture(0)
        yolo_net, yolo_output_layers = load_yolo()
        east_net = load_east()
        classes = ["person"]

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame, person_count = detect(frame, yolo_net, yolo_output_layers, east_net, classes, send_email=False)
            if frame is not None:
                # Display person count in the corner
                cv2.putText(frame, f'People Count: {person_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                stframe.image(frame, channels="BGR")
        cap.release()

    elif option == "Surveillance":
        stframe = st.empty()
        cap = cv2.VideoCapture(0)
        yolo_net, yolo_output_layers = load_yolo()
        east_net = load_east()
        classes = ["person"]

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame, person_count = detect(frame, yolo_net, yolo_output_layers, east_net, classes, send_email=True)
            if frame is not None:
                stframe.image(frame, channels="BGR")
        cap.release()

    elif option == "Ghost Protocol":
        st.sidebar.title("Upload Target Image")
        target_image_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if target_image_file is not None:
            target_img = Image.open(target_image_file)
            target_img = np.array(target_img)
            target_face_encodings = face_recognition.face_encodings(target_img)

            st.write(f"Number of faces detected in target image: {len(target_face_encodings)}")
            #st.write(f"Target face encodings: {target_face_encodings}")

            if target_face_encodings:
                known_face_encodings = target_face_encodings
                known_face_names = ["Target Person"]

                stframe = st.empty()
                cap = cv2.VideoCapture(0)
                yolo_net, yolo_output_layers = load_yolo()
                east_net = load_east()
                classes = ["person"]

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame, person_count = detect(frame, yolo_net, yolo_output_layers, east_net, classes, send_email=True, known_face_encodings=known_face_encodings, known_face_names=known_face_names, tolerance=0.6)
                    if frame is not None:
                        stframe.image(frame, channels="BGR")
                cap.release()
            else:
                st.error("No face detected in the uploaded target image.")

if __name__ == "__main__":
    main()
