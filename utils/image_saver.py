import os
import cv2
import numpy as np

class ImageSaver:
    def __init__(self, base_paths):
        """
        Initialize the ImageSaver with base paths.

        :param base_paths: List of base paths to search for card folders.
        """
        self.base_paths = base_paths

    def _locate_card_folder(self, card_name):
        """
        Locate the card folder within the base paths.

        :param card_name: Name of the card (used to locate its folder).
        :return: Path to the card folder.
        """
        for base_path in self.base_paths:
            card_folder_path = os.path.join(base_path, card_name)
            if os.path.exists(card_folder_path):
                return card_folder_path
        raise FileNotFoundError(f"Card folder '{card_name}' not found in the specified base paths.")

    def save_images(self, card_name, subdirectory_name, image_paths, base_name):
        """
        Save a list of images from paths to a subdirectory within the card folder.

        :param card_name: Name of the card (used to locate its folder).
        :param subdirectory_name: Name of the subdirectory to create within the card folder.
        :param image_paths: List of image file paths to save.
        :param base_name: Base name for the image files.
        """
        # Locate the card folder
        card_folder_path = self._locate_card_folder(card_name)

        # Create the subdirectory within the card folder
        subdirectory_path = os.path.join(card_folder_path, subdirectory_name)
        os.makedirs(subdirectory_path, exist_ok=True)

        # Save each image in the list
        for idx, image_path in enumerate(image_paths, start=1):
            file_name = f"{base_name}_{idx:02d}.jpg"  # Add sequential numbers like 01, 02, etc.
            file_path = os.path.join(subdirectory_path, file_name)

            # Read the image from the given path
            image = cv2.imread(image_path)
            if image is not None:
                cv2.imwrite(file_path, image)
            else:
                raise ValueError(f"Unable to read image from path: {image_path}")

        print(f"{len(image_paths)} images saved to {subdirectory_path}")


    def save_image_arrays(self, card_name, subdirectory_name, image_arrays, base_name):
        """
        Save a list of image arrays to a subdirectory within the card folder.

        :param card_name: Name of the card (used to locate its folder).
        :param subdirectory_name: Name of the subdirectory to create within the card folder.
        :param image_arrays: List of image arrays (numpy arrays) to save.
        :param base_name: Base name for the image files.
        """
        print("from save_image_arrays")
        # Locate the card folder
        card_folder_path = self._locate_card_folder(card_name)

        # Create the subdirectory within the card folder
        subdirectory_path = os.path.join(card_folder_path, subdirectory_name)
        os.makedirs(subdirectory_path, exist_ok=True)

        # Save each image array in the list
        for idx, image_array in enumerate(image_arrays, start=1):
            file_name = f"{base_name}_{idx:02d}.jpg"  # Add sequential numbers like 01, 02, etc.
            file_path = os.path.join(subdirectory_path, file_name)

            if isinstance(image_array, np.ndarray):
                # Ensure the image is in BGR format for proper saving
                if len(image_array.shape) == 2:  # Grayscale image
                    image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
                elif image_array.shape[2] == 3:  # Ensure it's in BGR format
                    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

                # Save the image
                cv2.imwrite(file_path, image_array)
            else:
                raise ValueError("Each item in image_arrays must be a numpy array.")

        print(f"{len(image_arrays)} images saved to {subdirectory_path}")
