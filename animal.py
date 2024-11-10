import tensorflow as tf
import cv2
import numpy as np

# Load MobileNetV2 model with ImageNet weights
model = tf.keras.applications.MobileNetV2(weights="imagenet")

# Preprocess the frame for MobileNetV2
def preprocess_image(frame):
    # Resize the frame to 224x224, which is the input size MobileNetV2 expects
    img = cv2.resize(frame, (224, 224))
    # Expand dimensions to make it compatible with model input
    img = np.expand_dims(img, axis=0)
    # Preprocess the image (normalize in the same way as MobileNetV2 training data)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img

# Get labels for predictions
def get_label(preds):
    # Decode the predictions to get human-readable labels
    labels = tf.keras.applications.mobilenet_v2.decode_predictions(preds)
    for label in labels[0]:
        # Filter for common animal classes by label name
        if "dog" in label[1] or "cat" in label[1] or "horse" in label[1] or "bird" in label[1] or "monkey" in label[1]:
            return label[1], label[2]  # Return label name and confidence
    return None, None  # If no animal detected, return None

# Access the camera
cap = cv2.VideoCapture(0)  # 0 is usually the default camera

while True:
    # Read frame
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess and predict
    img = preprocess_image(frame)
    preds = model.predict(img)
    label, confidence = get_label(preds)

    # Display result on the frame
    if label:
        text = f"{label}: {confidence:.2f}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Animal Detection & Classification ", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()

