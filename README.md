# Real-Time Mood Detection using Deep Learning

This repository contains a Python script that uses a pre-trained deep learning model to detect human moods in real-time via webcam. The model predicts one of several possible emotions by analyzing facial expressions from a video feed.

## Features

- Real-time facial emotion detection using your webcam.
- Supports a range of emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral.
- Easy-to-use Python script with OpenCV and TensorFlow integration.
- Simple preprocessing pipeline for consistent input to the model.

## Prerequisites

Before running the script, make sure you have the following installed:

- Python 3.7 or higher
- TensorFlow 2.x
- OpenCV 4.x
- Numpy
- Scikit Learn
- Keras
- Matplotlib

You can install the necessary Python packages using `pip`:


`pip install tensorflow opencv-python keras matplotlib numpy`

## Getting Started

1. **Clone the repository:**
    
        
    `git clone https://github.com/Chris-debuggs/Real-time_moodDetection-using-CV-.git cd mood-detection`
    
2. **Download or place your pre-trained model** in the root of the repository. The model file should be named `mood_model.keras` or update the script to point to your model file location.
    
3. **Run the script:**
    
        
    `python mood.py`
    
4. **Press 'q'** to exit the real-time mood detection window.
    

## How It Works

- The script captures frames from your webcam in real-time.
- Each frame is preprocessed: converted to grayscale, resized to 48x48 pixels, normalized, and reshaped to fit the model's input requirements.
- The preprocessed frame is passed through the deep learning model, which outputs probabilities for each emotion class.
- The emotion with the highest probability is displayed on the video feed in real-time.

## Model Details

The pre-trained model used in this script is designed to detect the following emotions:

- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

You can train your own model using datasets like FER2013 and integrate it with the script.

## Customization

- **Using a Different Model:** If you have a different model, update the path in the script where the model is loaded:
    
    python
        
    `model = load_model('path/to/your/model.keras')`
    
- **Modifying Emotions:** If your model predicts a different set of emotions, update the `emotions` list accordingly:
    
    python
        
    `emotions = ['Emotion1', 'Emotion2', ...]`
    

## Contribution

Feel free to fork this repository, submit issues, or contribute by opening pull requests. Any enhancements or bug fixes are welcome!

## License

This project is licensed under the MIT License.

## Acknowledgments

- [TensorFlow](https://www.tensorflow.org/)
- [OpenCV](https://opencv.org/)
- [Keras](https://keras.io/api/)
- [Scikit Learn](https://scikit-learn.org/stable/)
- [FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013?resource=download)
