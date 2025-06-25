"""
DISCLAIMER - AMINDAV PROPERTY

This software is the exclusive property of Amindav. All rights reserved.
Unauthorized copying, distribution, or modification of this code is strictly prohibited.
This module is designed for Trello integration in the bride and groom identification system.

SYSTEM OVERVIEW:
This module provides comprehensive Trello integration for project management.
It handles all Trello operations including:
1. Board and list management
2. Card movement between lists
3. File attachment uploads
4. Message writing to cards
5. Authentication and API communication

WORKFLOW:
1. Authenticates with Trello API using provided credentials
2. Locates the specified board for project management
3. Manages cards across different lists (IN, PROCESS, OUT, ERROR)
4. Uploads processed images and results to cards
5. Provides status updates and error messages

AUTHOR: Amindav Development Team
VERSION: 1.0
"""

from trello import TrelloClient
import io
import cv2
import os
import numpy as np

class TrelloManager:
    """
    A class to manage Trello operations, such as retrieving boards and lists,
    moving cards, and uploading attachments to cards.

    This class provides a high-level interface for all Trello operations needed
    by the bride and groom identification system. It handles authentication,
    board management, card operations, and file uploads.

    Attributes:
        client (TrelloClient): An authenticated Trello client for interacting with the Trello API.
        board (Board): The Trello board identified by the provided board name.
    """

    def __init__(self, api_key, api_secret, token, token_secret, board_name):
        """
        Initializes the TrelloManager instance by authenticating with Trello API
        and locating the specified board.

        This constructor sets up the Trello client with the provided credentials
        and validates that the specified board exists.

        Args:
            api_key (str): Trello API key for authentication.
            api_secret (str): Trello API secret for authentication.
            token (str): Trello user token for authentication.
            token_secret (str): Trello token secret for authentication.
            board_name (str): The name of the Trello board to interact with.

        Raises:
            ValueError: If the specified board is not found.
        """
        # Initialize Trello client with provided credentials
        self.client = TrelloClient(
            api_key=api_key,
            api_secret=api_secret,
            token=token,
            token_secret=token_secret
        )
        # Locate and validate the specified board
        self.board = self.get_board_by_name(board_name)
        if not self.board:
            raise ValueError(f"Board '{board_name}' not found.")

    def get_board_by_name(self, board_name):
        """
        Retrieves a Trello board by its name.

        Searches through all accessible boards to find the one with the
        specified name. This is used during initialization and can be
        called independently if needed.

        Args:
            board_name (str): The name of the board to find.

        Returns:
            Board: The board object if found, otherwise None.
        """
        boards = self.client.list_boards()
        for board in boards:
            if board.name == board_name:
                return board
        return None

    def get_cards_from_list(self, list_name="IN"):
        """
        Retrieves all cards from a specified list on the board.

        This method is used to get cards from different lists in the workflow:
        - "IN" list: New projects to process
        - "PROCESS" list: Projects currently being processed
        - "OUT" list: Successfully completed projects
        - "ERROR" list: Failed projects

        Args:
            list_name (str): The name of the list to retrieve cards from. Defaults to "IN".

        Returns:
            list: A list of Card objects in the specified list. Returns an empty list if the list is not found.
        """
        lists = self.board.list_lists()
        for trello_list in lists:
            if trello_list.name == list_name:
                return trello_list.list_cards()
        print(f"List '{list_name}' not found.")
        return []

    def move_card_to_list(self, card, target_list_name):
        """
        Moves a card to a specified list on the board.

        This method is crucial for the workflow management, allowing cards to
        move between different stages of processing (IN -> PROCESS -> OUT/ERROR).

        Args:
            card (Card): The Trello card to move.
            target_list_name (str): The name of the target list.

        Returns:
            str: The ID of the target list if the move is successful, otherwise None.
        """
        lists = self.board.list_lists()
        for trello_list in lists:
            if trello_list.name == target_list_name:
                card.change_list(trello_list.id)
                print(f"Moved card '{card.name}' to list '{target_list_name}'.")
                return trello_list.id
        print(f"List '{target_list_name}' not found.")
        return None

    def upload_attachments_to_card(self, card, image_buffers, tag):
        """
        Uploads image attachments to a specified Trello card.

        This method processes image buffers (numpy arrays) and uploads them
        as attachments to Trello cards. It's used for uploading processed
        images like family photos and face verification results.

        Args:
            card (Card): The Trello card to attach images to.
            image_buffers (list): A list of image buffers (numpy arrays) to attach.
            tag (str): A tag to include in the uploaded file names for identification.

        Raises:
            Exception: If any error occurs during the upload process.
        """
        try:
            for i, image_buffer in enumerate(image_buffers):
                # Convert numpy array to image bytes for upload
                image_bytes = cv2.imencode('.jpg', image_buffer)[1].tobytes()
                image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
                # Convert to RGB for Mediapipe compatibility
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                success, img_encoded = cv2.imencode(".jpg", rgb_image)
                image_file = io.BytesIO(img_encoded.tobytes())

                # Generate unique filename for the attachment
                image_file.name = f"{card.name.replace('+', '')}_{tag}_{i+1:03}.jpg"

                # Attach the image to the card
                card.attach(
                    name=image_file.name,
                    file=image_file,
                    mimeType="image/jpeg"
                )
                print(f"Uploaded attachment '{image_file.name}' to card '{card.name}'.")
        except Exception as e:
            print(f"Error uploading attachments to card '{card.name}': {e}")

    def upload_attachments_to_card_dir(self, card, image_paths, tag):
        """
        Uploads image attachments to a specified Trello card from file paths.

        This method uploads images that have been saved to disk (typically
        from the face verification API) to Trello cards. It's used for
        uploading verified face images.

        Args:
            card (Card): The Trello card to attach images to.
            image_paths (list): A list of image file paths to attach.
            tag (str): A tag to include in the uploaded file names for identification.

        Raises:
            Exception: If any error occurs during the upload process.
        """
        try:
            for i, image_path in enumerate(image_paths):
                if not os.path.exists(image_path):
                    print(f"File not found: {image_path}")
                    continue
                
                # Generate a unique name for the file
                file_name = f"{card.name}_{tag}_{i+1:03}.jpg"
                
                # Attach the image to the card
                with open(image_path, "rb") as image_file:
                    card.attach(
                        name=file_name,
                        file=image_file,
                        mimeType="image/jpeg"
                    )
                print(f"Uploaded attachment '{file_name}' to card '{card.name}'.")
        except Exception as e:
            print(f"Error uploading attachments to card '{card.name}': {e}")

    def write_message_to_card(self, card, message):
        """
        Adds a message to the description or as a comment on a specified Trello card.

        This method is used to provide status updates, error messages, and
        processing information to cards. It helps track the progress and
        issues with each project.

        Args:
            card (Card): The Trello card to which the message will be added.
            message (str): The message to write on the card.

        Returns:
            None
        """
        try:
            # Add the message as a comment to the card
            card.comment(message)
            print(f"Message added to card '{card.name}': {message}")
        except Exception as e:
            print(f"Error adding message to card '{card.name}': {e}")
