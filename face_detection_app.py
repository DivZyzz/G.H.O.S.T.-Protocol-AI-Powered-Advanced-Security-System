# Import required libraries
import streamlit as st
import cv2
from PIL import Image
import numpy as np

# Function to detect faces in an image
def detect_faces(image):
    # Load the OpenCV face detection classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    return faces

# Streamlit app
def main():
    st.title("Face Detection App")

    # Upload image
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Detect faces in the uploaded image
        faces = detect_faces(np.array(image))

        if len(faces) == 0:
            st.write("No faces detected in the uploaded image.")
        else:
            st.write(f"{len(faces)} face(s) detected in the uploaded image.")

            # Access the webcam
            video_capture = cv2.VideoCapture(0)

            # Placeholder for the video feed
            placeholder = st.empty()

            # Run the app
            while True:
                # Capture frame-by-frame
                ret, frame = video_capture.read()

                # Convert the frame to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Detect faces
                result_frame = rgb_frame.copy()
                for (x, y, w, h) in faces:
                    cv2.rectangle(result_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

                # Display the result
                placeholder.image(result_frame, channels="RGB", use_column_width=True)

                # Check if 'q' is pressed to exit the loop
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Release the video capture object and close all windows
            video_capture.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
