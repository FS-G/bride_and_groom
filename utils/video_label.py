import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from PIL import Image
from moviepy.editor import VideoFileClip


class VideoLabel:
    def __init__(self, model_path, img_size=(224, 224), num_classes=3, class_labels=None, n_frames=10):
        """
        Initialize the VideoLabel class.

        Args:
            model_path (str): Path to the trained model weights.
            img_size (tuple): Image size for resizing frames (default: (224, 224)).
            num_classes (int): Number of classes (default: 3).
            class_labels (dict): Dictionary mapping class indices to labels.
            n_frames (int): Number of frames to process (default: 10).
        """
        self.model_path = model_path
        self.img_size = img_size
        self.num_classes = num_classes
        self.class_labels = class_labels or {0: "ceremony", 1: "dance", 2: "other"}
        self.n_frames = n_frames
        self.input_shape = (*img_size, 3)
        self.model = self._load_model()

    def _create_cnn_model(self):
        """
        Create the CNN model structure.

        Returns:
            model: Compiled CNN model.
        """
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=self.input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(self.num_classes, activation="softmax"),
        ])
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        return model

    def _load_model(self):
        """
        Load the model with weights.

        Returns:
            model: The loaded model with pre-trained weights.
        """
        model = self._create_cnn_model()
        model.load_weights(self.model_path)
        print("Model weights loaded successfully.")
        return model

    def _predict_frame(self, frame):
        """
        Predict the class of a single frame and return confidence scores.

        Args:
            frame: A numpy array of the frame.

        Returns:
            tuple: Predicted class label and a dictionary of confidence scores for all classes.
        """
        frame_array = frame / 255.0  # Normalize pixel values
        frame_array = np.expand_dims(frame_array, axis=0)  # Add batch dimension
        
        # Get prediction probabilities for each class
        pred = self.model.predict(frame_array, verbose=0)
        pred = pred[0]  # Assuming `pred` is a batch of probabilities, take the first element
        
        # Find the predicted class index
        predicted_class = np.argmax(pred)
        
        # Map class indices to labels
        confidence_scores = {self.class_labels[i]: prob for i, prob in enumerate(pred)}
        predicted_label = self.class_labels.get(predicted_class, "other")  # Default to "other"
        
        return predicted_label, confidence_scores

    def label_video(self, video_path):
        """
        Process the video and predict labels for frames, including confidence scores.

        Args:
            video_path (str): Path to the video file.

        Returns:
            tuple: Counts of 'dance', 'ceremony', and 'other' labels.
        """
        clip = VideoFileClip(video_path)
        duration = clip.duration  # Duration of the video in seconds
        timestamps = np.linspace(0, duration, self.n_frames, endpoint=False)
        
        dance, ceremony, other = 0, 0, 0
        for t in timestamps:
            frame = clip.get_frame(t)
            frame = Image.fromarray(frame).convert("RGB").resize(self.img_size)
            frame_array = np.array(frame)
            
            # Predict label and get confidence scores
            label, confidences = self._predict_frame(frame_array)
            print(f"Timestamp {t:.2f}s: Label = {label}, Confidence Scores = {confidences}")
            
            if label == "dance":
                dance += 1
            elif label == "ceremony":
                ceremony += 1
            else:
                other += 1

        clip.close()
        return dance, ceremony, other
