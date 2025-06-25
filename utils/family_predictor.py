import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras import layers, models

class FamilyPredictor:
    def __init__(self, model_weights_path, img_size=(224, 224)):
        """
        Initialize the FamilyPredictor class with the model weights and image size.

        :param model_weights_path: Path to the saved model weights.
        :param img_size: Tuple specifying the image size for resizing.
        """
        self.img_size = img_size
        self.model = self._load_model(model_weights_path)

    def _load_model(self, model_weights_path):
        """
        Load the trained CNN model and its weights.

        :param model_weights_path: Path to the saved model weights.
        :return: A compiled CNN model with loaded weights.
        """
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=(*self.img_size, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(1, activation="sigmoid"),  # Sigmoid for binary classification
        ])
        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",  # Binary crossentropy for binary classification
            metrics=["accuracy"],
        )
        model.load_weights(model_weights_path)
        return model

    def _preprocess_image(self, image_array):
        """
        Preprocess a single image array for prediction.

        :param image_array: Numpy array representing the image.
        :return: Preprocessed image array.
        """
        img = Image.fromarray(np.uint8(image_array))  # Ensure it's in uint8 format
        img = img.resize(self.img_size)
        img_array = np.array(img) / 255.0  # Normalize pixel values
        return np.expand_dims(img_array, axis=0)  # Add batch dimension

    def predict_top_family_images(self, image_arrays, top_n=5):
        """
        Predict the "family" probabilities for a list of images and return the top N.

        :param image_arrays: List of image arrays (numpy arrays).
        :param top_n: Number of top images with the highest "family" probabilities to return.
        :return: List of top N image arrays with the highest "family" probabilities.
        """
        probabilities = []
        for img_array in image_arrays:
            preprocessed_img = self._preprocess_image(img_array)
            prediction = self.model.predict(preprocessed_img)[0][0]  # Get probability for "family" class
            probabilities.append((prediction, img_array))
        
        # Sort by probabilities in descending order and select top N
        top_images = sorted(probabilities, key=lambda x: x[0], reverse=True)[:top_n]
        return [img[1] for img in top_images]  # Return the image arrays of top N


