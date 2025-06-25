"""
DISCLAIMER - AMINDAV PROPERTY

This software is the exclusive property of Amindav. All rights reserved.
Unauthorized copying, distribution, or modification of this code is strictly prohibited.
This system is designed for automated bride and groom identification in wedding videos.

SYSTEM OVERVIEW:
This is the main entry point for the Bride and Groom Identification System.
The system processes wedding videos to:
1. Categorize videos (ceremony, dance/party, other)
2. Detect tripod-stabilized videos
3. Extract and identify bride and groom faces from ceremony videos
4. Generate family photos with the best bride-groom interactions
5. Upload results to Trello for project management

WORKFLOW:
1. Monitors Trello "IN" list for new project cards
2. Moves cards to "PROCESS" list during processing
3. Finds videos matching the project criteria
4. Processes videos through multiple AI models
5. Uploads results and moves cards to "OUT" or "ERROR" lists

AUTHOR: Amindav Development Team
VERSION: 1.0
"""

# import external libraries
import json
import os
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from PIL import Image
from ultralytics import YOLO
import uuid
import requests


# import internal modules
from utils.trello_manager import TrelloManager
from utils.video_label import VideoLabel
from utils.video_finder import VideoFinder
from utils.textfile_writer import TextFileWriter
from utils.bride_groom_extractor import BrideGroomExtractor
from utils.tripod_detector import TripodDetector
from utils.image_saver import ImageSaver
from utils.family_predictor import FamilyPredictor


# Trello API credentials for project management integration
# These credentials allow the system to interact with the Trello board
API_KEY = "a829f8cc90164eb242f21d284e658c3a"
API_SECRET = "96e5c5500af5b1a21fc77d79c1d25371b5863835cefcf18e3904de5ad0ea9254"
TOKEN = "18dd204ce71821f328fac9d6be5bb86d1e4f86a86cc95e7534eb6f2789d86a85"
TOKEN_SECRET = "bb8b51dba51bd6f9431dba6f765f6d85" 


# Load configuration parameters from JSON file
# This file contains all the system settings and model IDs
with open('parameters.json') as file:
    parameters = json.load(file)


# Extract configuration parameters for system operation
base_paths = parameters['base_paths']  # Directories to search for video files
board_name = parameters['trello_board']  # Trello board name for project management
file_length = parameters['file_length']  # Minimum video length threshold (seconds)
frames_per_video = parameters['frames_per_video']  # Number of frames to extract per video
model_id = parameters['model_id_video']  # ID of the video classification model
model_id_bg = parameters["model_id_bg"]  # ID of the bride/groom detection model
model_id_face = parameters["model_id_face"]  # ID of the face extraction model
in_list = parameters["in_list"]  # Trello list for incoming projects
process_list = parameters["process_list"]  # Trello list for projects being processed
out_list = parameters["out_list"]  # Trello list for completed projects
error_list = parameters["error_list"]  # Trello list for failed projects
motion_threshold = parameters["motion_threshold"]  # Threshold for tripod detection


# Initialize all system components and AI models
# Each component handles a specific aspect of the video processing pipeline
trello_manager = TrelloManager(api_key=API_KEY, api_secret=API_SECRET, token=TOKEN, token_secret=TOKEN_SECRET,board_name=board_name)
video_finder = VideoFinder(base_paths)
video_labeler = VideoLabel(model_path=f".\models\cnn_model_weights{model_id}.weights.h5",img_size=(224, 224),num_classes=3,class_labels={0: "ceremony", 1: "dance", 2: "other"},n_frames=frames_per_video)
writer = TextFileWriter(base_paths)
yolo_model_base = YOLO(f"models_yolo/base{model_id_bg}.pt")
yolo_model_face = YOLO(f"models_yolo/face{model_id_face}.pt")
extractor = BrideGroomExtractor(yolo_model_base, yolo_model_face)
detector = TripodDetector(motion_threshold=motion_threshold)
saver = ImageSaver(base_paths)
predictor = FamilyPredictor("models/cnn_model_family_weights1.weights.h5")


########################################
# # optional - should be commented
# def save_faces(bride_faces, card, tag):
#     # Ensure directory exists
#     directory_name = card.name
#     if not os.path.exists(directory_name):
#         os.makedirs(directory_name)
#     # Save each image with a unique name
#     for i, image in enumerate(bride_faces):
#         file_name = f"{tag}_{uuid.uuid4().hex}.jpg"  # Generate a unique filename
#         file_path = os.path.join(directory_name, file_name)
#         cv2.imwrite(file_path, image)  # Save the image
#         print(f"Saved image {i+1} as {file_name} in {directory_name}")
##########################################


def send_image(image_list, url="http://127.0.0.1:8000/process_face/"):
    """
    Sends a list of images to the face processing API for verification and grouping.
    
    This function communicates with the parallel API service that uses DeepFace
    to verify and group similar faces, ensuring we have the best quality images
    for bride and groom identification.
    
    Args:
        image_list (list): List of NumPy arrays representing face images
        url (str): URL of the face processing API endpoint
        
    Returns:
        dict: Response from the API containing grouped face information
        None: If the request fails
        
    Raises:
        ValueError: If image_list is not a list or contains invalid images
    """
    if not isinstance(image_list, list):
        raise ValueError("Expected 'image_list' to be a list of NumPy arrays.")
    
    # Prepare the files payload for the API request
    files = []
    for i, image_array in enumerate(image_list):
        if isinstance(image_array, np.ndarray):
            # Ensure the image is in RGB format for optimal processing
            image_bytes = cv2.imencode('.jpg', image_array)[1].tobytes()
            image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            # Convert to RGB for Mediapipe compatibility
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            success, img_encoded = cv2.imencode(".jpg", rgb_image)
            if not success:
                raise ValueError(f"Failed to encode image at index {i}.")
            files.append(("files", ("image.jpg", img_encoded.tobytes(), "image/jpeg")))
        else:
            raise ValueError(f"Element at index {i} is not a valid NumPy array.")
    
    # Send POST request to the face processing API
    try:
        response = requests.post(url, files=files, timeout=6000)
        response.raise_for_status()
        return response.json()  # Expecting JSON response with grouped faces
    except requests.exceptions.RequestException as e:
        print(f"Error during request: {e}")
        return None


# Main processing loop - Get cards from the "IN" list to process
cards = trello_manager.get_cards_from_list(in_list)
################################
# # for debugging only
# class Card:
#     def __init__(self, name):
#         self.name = name

# # Original list
# cards = [ "_+HAPPYD+_Gili_Omer", "_+DAYO+_Shaked_Shlomi", "_+DORH+_Bar_Raz"]

# # Convert the list into a list of Card objects
# cards = [Card(name) for name in cards]
###############################
print(cards)
if not cards:
    print('No card in the in list')
# if cards are in the "IN" list  
else:
    # Process each card in the IN list sequentially
    for card in cards:

        # Check for stuck cards in PROCESS list and move them to ERROR
        # This prevents cards from getting stuck in processing state
        process_cards = trello_manager.get_cards_from_list(process_list)
        if len(process_cards) > 1:
            for i in range(len(process_cards)):
                process_card = process_cards[i]

                trello_manager.write_message_to_card(process_card, "This card was stuck in the 'PROCESS' list")

                # Move a card to ERROR list
                trello_manager.move_card_to_list(process_card, error_list)


        # Move current card to PROCESS list to indicate processing has started
        trello_manager.move_card_to_list(card, process_list)

        # Find videos paths for the trello card qualifying the max_length condition
        # Videos are sorted in descending order by length
        matching_videos = video_finder.find_videos(card.name, file_length = file_length) # file length threshold for now is set to 0
        print("Matching Videos:", matching_videos)

        # Error handling: If no qualifying videos are found, move card to error list
        if matching_videos == "NULL":
            trello_manager.move_card_to_list(card, error_list)
            trello_manager.write_message_to_card(card, "No matching directory found")
            print(f'{card.name} moved to ERROR list')

        elif not matching_videos:
            trello_manager.move_card_to_list(card, error_list)
            trello_manager.write_message_to_card(card, "no video longer than file_length in JSON.")
            print(f'{card.name} moved to ERROR list')
        else:

            # Detect videos filmed with tripod and write them to text file
            # Tripod videos are typically more stable and better for face extraction
            tripod_videos = detector.find_tripod_videos(matching_videos)
            writer.write_text_file_tripod(card, tripod_videos)

            # Video categorization phase
            # Create a dictionary to hold categorized video file names
            categorized_videos = {"Ceremony": [], "Dance/Party": [], "other": []}
            ceremony_list = []

            # Loop through each video path and classify them
            for current_video_path in matching_videos:
                # Use AI model to determine video category (ceremony, dance, or other)
                dance_count, ceremony_count, other_count = video_labeler.label_video(current_video_path)
                # Determine the category with the highest count
                max_var = max(("dance", dance_count), ("ceremony", ceremony_count), ("other", other_count), key=lambda x: x[1])[0]     
                # Categorize the video based on the detected label
   
                if max_var == "dance":
                    categorized_videos["Dance/Party"].append(os.path.basename(current_video_path))
                elif max_var == "ceremony":
                    categorized_videos["Ceremony"].append(os.path.basename(current_video_path))
                    ceremony_list.append(current_video_path)

                elif max_var == "other":
                    categorized_videos["other"].append(os.path.basename(current_video_path))

            # Create a text file documenting the video categories for this project
            writer.write_text_file(card, categorized_videos)

            # Bride and Groom Face Extraction Phase
            # Only process if ceremony videos were detected
            if len(ceremony_list) > 0:
                bride_faces = []
                # Process each ceremony video to extract faces
                for current_ceremony_path in ceremony_list:

                    # Confidence levels for face detection - start high and reduce if needed
                    confidence_list = [0.7, 0.65, 0.62, 0.6, 0.58, 0.56]
                    
                    groom_faces = []
                    bride_and_groom_frame_list = []
                    
                    # Try different confidence levels to get optimal face detection
                    for confidence in confidence_list:
                        
                        # Open video and process frame by frame
                        with VideoFileClip(current_ceremony_path) as clip:
                            
                            duration = clip.duration
                            # Extract frames evenly distributed throughout the video
                            timestamps = np.linspace(0, duration, frames_per_video * 10, endpoint=False)
                            
                            for t in timestamps:
                                print(f'Number of brides : {len(bride_faces)} and number of grooms {len(groom_faces)}')
                                # Stop if we have enough faces for both bride and groom
                                if len(bride_faces) > 11 and len(groom_faces) > 11:
                                    break
                                # Skip to next video if bride detection is poor but groom detection is excessive
                                if len(bride_faces) < 5 and len(groom_faces) > 100:
                                    print("Going to the next ceremony video if available as the bride's face is not clearly visibly in this video")
                                    break
                                
                                # Extract frame at current timestamp
                                frame = clip.get_frame(t)
                                frame_array = np.array(frame)
                                
                                # Use the extractor to identify bride and groom faces
                                cropped_persons, bride_and_groom_frame = extractor.extract_faces(frame_array, confidence = confidence, card_name = card.name)
                                
                                # Store detected faces
                                if 'bride' in cropped_persons and cropped_persons['bride'] is not None:
                                    bride_faces.append(cropped_persons['bride']['image'])
                                if 'groom' in cropped_persons and cropped_persons['groom'] is not None:
                                    groom_faces.append(cropped_persons['groom']['image'])
                                if bride_and_groom_frame is not None:
                                    bride_and_groom_frame_list.append(bride_and_groom_frame)

                        # If we have enough faces, stop processing this video
                        if len(bride_faces) > 10 and len(groom_faces) > 10:
                            break   
                        print(f'reducing the confidence from {confidence}')            

                    # If we have enough faces from this video, stop processing more videos
                    if len(bride_faces) > 10 and len(groom_faces) > 10:
                        break   

                # Face Verification and Upload Phase
                # Send identified faces for verification and upload to Trello
                ERROR_FLAG = False
                
                # Process bride faces
                if len(bride_faces) > 5:
                    # Limit to 12 best images for processing efficiency
                    if len(bride_faces) > 12:
                        bride_faces = bride_faces[:12]

                    print("Deep face analyzing brides")
                    # Send bride faces to API for verification and grouping
                    face_path_bride = send_image(bride_faces)
                    # Save verified bride images locally
                    saver.save_images(card.name, "PROFILE_PICTURES", face_path_bride["group"], "BRIDE")
                    # Upload bride images to Trello card
                    trello_manager.upload_attachments_to_card_dir(card, face_path_bride["group"], "BRIDE")

                else:
                    print(f"less bride images {len(bride_faces)}")
                    trello_manager.write_message_to_card(card, f"less bride images {len(bride_faces)}")
                    ERROR_FLAG = True

                # Process groom faces
                if len(groom_faces) > 5:
                    # Limit to 12 best images for processing efficiency
                    if len(groom_faces) > 12:
                        groom_faces = groom_faces[:12]

                    print("Deep face analyzing grooms")
                    # Send groom faces to API for verification and grouping
                    face_path_groom = send_image(groom_faces)
                    # Save verified groom images locally
                    saver.save_images(card.name, "PROFILE_PICTURES", face_path_groom["group"], "GROOM")
                    # Upload groom images to Trello card
                    trello_manager.upload_attachments_to_card_dir(card, face_path_groom["group"], "GROOM")
                else:
                    print(f"less groom images {len(groom_faces)}")
                    trello_manager.write_message_to_card(card, f"less groom images {len(groom_faces)}")
                    ERROR_FLAG = True

                # Final processing decision based on face extraction success
                if ERROR_FLAG:
                    # If face extraction failed, move card to error list
                    print("The card moved to the error list")
                    trello_manager.move_card_to_list(card, error_list)
                    
                else:
                    # Generate family photos from bride-groom interaction frames
                    print("saving the family pics")
                    print("length of bride and groom list", len(bride_and_groom_frame_list))
                    # Use AI to predict the best family photo moments
                    top_images = predictor.predict_top_family_images(bride_and_groom_frame_list)
                    # Save family images locally
                    saver.save_image_arrays(card.name, "FAMILY_PICTURES", top_images, f"FAMILY_{card.name}")
                    # Upload family images to Trello card
                    trello_manager.upload_attachments_to_card(card, top_images, "FAMILY")
                    # Move card to OUT list indicating successful completion
                    print("The card moved to the out list")
                    trello_manager.move_card_to_list(card, out_list)

            else:
                # No ceremony videos found - move to error list
                trello_manager.move_card_to_list(card, error_list)
                trello_manager.write_message_to_card(card, "No ceremony video found")
                print(f'{card.name} moved to ERROR list')
                
                

































