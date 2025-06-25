import os

class TextFileWriter:
    def __init__(self, base_paths):
        """
        Initialize the TextFileWriter with base paths.
        
        :param base_paths: List of base paths to search for card folders.
        """
        self.base_paths = base_paths

    def write_text_file(self, card, categorized_videos):
        """
        Write a text file in the root folder of the card's name.
        
        :param card: An object with a `.name` attribute representing the card's name.
        :param categorized_videos: Dictionary with categories as keys and file lists as values.
        """
        # Locate the base folder where the card's name exists
        for base_path in self.base_paths:
            card_folder_path = os.path.join(base_path, card.name)
            if os.path.exists(card_folder_path):
                break
        else:
            raise FileNotFoundError(f"Card folder '{card.name}' not found in the specified base paths.")

        # Construct the path to save the text file in the card's root folder
        text_file_path = os.path.join(card_folder_path, f"{card.name}_files_classification.txt")

        # Write the categorized videos to the text file
        # with open(text_file_path, "w") as f:
        with open(text_file_path, "w", encoding="utf-8") as f:
            for category, files in categorized_videos.items():
                f.write(f"{category}:\n")
                for file_name in files:
                    f.write(f"{file_name}\n")
                f.write("\n")  # Add a blank line between categories

        print(f"File classification written to {text_file_path}")

    def write_text_file_tripod(self, card, video_list):
        """
        Write a plain list of videos to a text file in the root folder of the card's name.

        :param card: An object with a `.name` attribute representing the card's name.
        :param video_list: List of video file names to write to the text file.
        """
        # Locate the base folder where the card's name exists
        for base_path in self.base_paths:
            card_folder_path = os.path.join(base_path, card.name)
            if os.path.exists(card_folder_path):
                break
        else:
            raise FileNotFoundError(f"Card folder '{card.name}' not found in the specified base paths.")

        # Construct the path to save the text file in the card's root folder
        text_file_path = os.path.join(card_folder_path, f"{card.name}_files_tripod.txt")

        # Write the plain list of video files to the text file
        with open(text_file_path, "w") as f:
            for file_name in video_list:
                f.write(f"{file_name}\n")

        print(f"Video list written to {text_file_path}")
