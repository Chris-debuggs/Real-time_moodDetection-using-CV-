import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('C:/Users/cnevi/Downloads/ComVi/mood_model.keras')

# Define the list of emotions (adjust according to your model's output)
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess the frame for mood detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    resized_frame = cv2.resize(gray, (48, 48))  # Resize to 48x48, the input size expected by the model
    normalized_frame = resized_frame / 255.0  # Normalize the image to [0, 1]
    reshaped_frame = normalized_frame.reshape(1, 48, 48, 1)  # Reshape to match model input shape

    # Predict the mood
    prediction = model.predict(reshaped_frame)
    mood_index = np.argmax(prediction)  # Get the index of the highest confidence value
    mood = emotions[mood_index]  # Get the corresponding emotion label

    # Display the mood on the frame
    cv2.putText(frame, mood, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the frame with mood detection
    cv2.imshow('Mood Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
