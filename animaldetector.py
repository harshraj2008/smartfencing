import cv2
from keras.models import model_from_json
import numpy as np

# Load the pre-trained model
json_file = open("Animal/training_hist.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)



# Load the face cascade classifier
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Define labels for animals
labels = {0: 'bear', 1: 'chinkara', 2: 'elephant', 3: 'lion', 4: 'peacock', 5: 'pig', 6: 'sheep', 7: 'tiger'}

# Function to preprocess input image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Initialize webcam
webcam = cv2.VideoCapture(0)

# Skip frames to reduce computational load
frame_skip = 5
frame_count = 0

# Main loop for real-time animal detection
while True:
    # Read a frame from the webcam
    ret, im = webcam.read()
    if not ret:
        break

    # Increment frame count and skip frames if necessary
    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    # Convert the frame to grayscale
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Iterate over detected faces
    for (x, y, w, h) in faces:
        # Extract the face region
        face_image = gray[y:y+h, x:x+w]

        # Resize the face image to match the model's input size
        face_image = cv2.resize(face_image, (48, 48))

        # Preprocess the face image
        img = extract_features(face_image)

        # Predict animal using the pre-trained model
        pred = model.predict(img)

        # Get the predicted animal label and confidence level
        prediction_label = labels[pred.argmax()]
        confidence_level = np.max(pred)

        # Draw rectangle around the face
        cv2.rectangle(im, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Annotate the frame with predicted animal and confidence level
        text = f'{prediction_label}: {confidence_level:.2f}'
        cv2.putText(im, text, (x-10, y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))

    # Display the annotated frame
    cv2.imshow("Output", im)

    # Wait for key press
    key = cv2.waitKey(1) & 0xFF

    # Break the loop if 'q' is pressed
    if key == ord('q'):
        break

# Release the webcam and close all OpenCV windows
webcam.release()
cv2.destroyAllWindows()
