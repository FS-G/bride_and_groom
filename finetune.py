"""
DISCLAIMER - AMINDAV PROPERTY

This software is the exclusive property of Amindav. All rights reserved.
Unauthorized copying, distribution, or modification of this code is strictly prohibited.
This system is designed for fine-tuning AI models used in the bride and groom identification system.

SYSTEM OVERVIEW:
This script performs two main tasks:
1.  Frame Extraction: Extracts frames from specified video files and saves them
    into categorized folders. The video sources and target categories are defined
    in the 'finetune_parameters.json' file.
2.  Model Fine-Tuning: Uses the extracted frames to fine-tune a pre-existing
    Convolutional Neural Network (CNN) model. It dynamically finds the latest
    model weights, trains on the new data, and saves the newly fine-tuned
    weights with an incremented version number.
3.  Parameter Update: After successful training, it updates the 'parameters.json'
    file with the new model version ID.

WORKFLOW:
1. Loads video targets from finetune_parameters.json
2. Extracts frames from specified videos into categorized folders
3. Finds the latest model weights for fine-tuning
4. Trains the model on new data with frozen early layers
5. Saves new model weights with incremented version
6. Updates parameters.json with new model ID

AUTHOR: Amindav Development Team
VERSION: 1.0
"""
import json
import os
import re
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from moviepy.editor import VideoFileClip
from PIL import Image
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- Configuration Constants ---

# Model & Training Hyperparameters
# These parameters control the CNN architecture and training process
IMG_SIZE: Tuple[int, int] = (224, 224)  # Standard input size for the CNN model
BATCH_SIZE: int = 10  # Number of samples per training batch
NUM_CLASSES: int = 3  # Corresponds to the number of subfolders in DATA_DIR (ceremony, dance, other)
LEARNING_RATE: float = 1e-4  # Learning rate for fine-tuning (lower than initial training)
EPOCHS: int = 5  # Number of training epochs
VALIDATION_SPLIT: float = 0.01  # 1% of data for validation, as in original script

# File and Directory Paths
CONFIG_FILE: Path = Path("finetune_parameters.json")  # Configuration file with video targets
PARAMS_JSON_FILE: Path = Path("parameters.json")  # Main parameters file to update after training
DATA_DIR: Path = Path("./finetune/data")  # Directory for extracted training data
MODEL_DIR: Path = Path("./model")  # Directory for storing model weights
MODEL_WEIGHTS_BASENAME: str = "cnn_model_weights"  # Base name for model weight files


# --- Helper Functions for Configuration and Pathing ---

def get_model_version_paths(model_dir: Path, basename: str) -> Tuple[Optional[Path], Path]:
    """
    Finds the latest model weights file and determines the path for the new one.

    Searches for files matching 'basename{version}.weights.h5', finds the highest
    version number, and returns its path along with the path for the next version.
    This ensures automatic versioning of model weights.

    Args:
        model_dir (Path): The directory where model weights are stored.
        basename (str): The base name for the weight files.

    Returns:
        A tuple containing:
        - Path to the latest existing weights file (or None if none found).
        - Path for the new weights file with an incremented version.
    """
    model_dir.mkdir(exist_ok=True)
    version_regex = re.compile(rf"{re.escape(basename)}(\d+)\.weights\.h5")
    
    latest_version = -1
    for f in model_dir.iterdir():
        match = version_regex.match(f.name)
        if match:
            version = int(match.group(1))
            if version > latest_version:
                latest_version = version

    if latest_version == -1:
        print("INFO: No previous model weights found. Will train a new model from scratch.")
        pretrained_path = None
    else:
        pretrained_path = model_dir / f"{basename}{latest_version}.weights.h5"
        print(f"INFO: Found latest model weights at version {latest_version}.")

    new_version = latest_version + 1
    new_weights_path = model_dir / f"{basename}{new_version}.weights.h5"
    
    return pretrained_path, new_weights_path

def load_video_targets_from_config(config_path: Path) -> List[Dict[str, str]]:
    """
    Loads video processing targets from a JSON configuration file.

    This function reads the finetune_parameters.json file to get the list of
    videos to process and their target categories for training data generation.

    Args:
        config_path (Path): The path to the JSON configuration file.

    Returns:
        A list of dictionaries, each containing a 'video_path' and 'target_folder'.
        
    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the JSON is malformed or has missing keys.
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        video_targets = config.get("video_targets")
        if not video_targets or not isinstance(video_targets, list):
            raise ValueError("Config file must contain a 'video_targets' list.")
            
        return video_targets
    except FileNotFoundError:
        print(f"ERROR: Configuration file not found at '{config_path}'")
        raise
    except json.JSONDecodeError:
        print(f"ERROR: Could not decode JSON from '{config_path}'. Check for syntax errors.")
        raise

def update_model_id_in_params(params_path: Path, new_weights_path: Path, basename: str):
    """
    Updates the 'model_id_video' in the parameters.json file with the new version.

    After successful model training, this function automatically updates the
    main parameters file so the system will use the newly trained model.

    Args:
        params_path (Path): Path to the parameters.json file.
        new_weights_path (Path): Path to the newly saved model weights.
        basename (str): The base name of the model weights file (e.g., 'cnn_model_weights').
    """
    print(f"\n--- Attempting to update '{params_path}' ---")

    # 1. Extract version number from the new weights path
    version_regex = re.compile(rf"{re.escape(basename)}(\d+)\.weights\.h5")
    match = version_regex.search(new_weights_path.name)
    
    if not match:
        print(f"ERROR: Could not extract version number from '{new_weights_path.name}'. Cannot update parameters file.")
        return

    new_model_version = int(match.group(1))
    print(f"INFO: New model version identified as: {new_model_version}")

    # 2. Read, update, and write the parameters.json file
    try:
        # Read the existing data, preserving all content including comments
        with open(params_path, 'r') as f:
            params_data = json.load(f)

        # Update the specific field
        params_data["model_id_video"] = new_model_version

        # Write the modified data back to the file with nice formatting
        with open(params_path, 'w') as f:
            json.dump(params_data, f, indent=4)
        
        print(f"✅ Successfully updated 'model_id_video' in '{params_path}' to {new_model_version}.")

    except FileNotFoundError:
        print(f"ERROR: Parameters file not found at '{params_path}'. Update failed.")
    except json.JSONDecodeError:
        print(f"ERROR: Could not decode JSON from '{params_path}'. Update failed.")
    except KeyError:
        print(f"ERROR: 'model_id_video' key not found in '{params_path}'. Update failed.")
    except Exception as e:
        print(f"ERROR: An unexpected error occurred while updating '{params_path}': {e}")

# --- Core Logic Functions (Unchanged) ---

def extract_and_save_frames(video_path: str, output_dir: Path, n_frames: int = 50):
    """
    Extracts, resizes, and saves evenly spaced frames from a video.

    This function processes a video file to extract training frames for the CNN model.
    It extracts frames evenly distributed throughout the video duration and resizes
    them to the standard input size for the model.

    Args:
        video_path (str): Path to the video file.
        output_dir (Path): Directory where the frames will be saved.
        n_frames (int): The number of frames to extract.
    """
    print(f"Processing video: {video_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        with VideoFileClip(video_path) as clip:
            duration = clip.duration
            if duration <= 0:
                print(f"WARNING: Video has zero or negative duration. Skipping: {video_path}")
                return
            
            # Extract frames evenly distributed throughout the video
            timestamps = np.linspace(0, duration, n_frames, endpoint=False)
            for t in timestamps:
                frame = clip.get_frame(t)
                frame_image = Image.fromarray(frame)

                # Resize image while maintaining aspect ratio
                orig_width, orig_height = frame_image.size
                new_height = IMG_SIZE[0]
                new_width = int(orig_width * new_height / orig_height)
                resized_image = frame_image.resize((new_width, new_height), Image.LANCZOS)

                # Save with unique filename to prevent conflicts
                frame_path = output_dir / f"{uuid.uuid4().hex}.jpg"
                resized_image.save(frame_path)
        print(f"-> Finished. Saved {n_frames} frames to '{output_dir}'.")
    except Exception as e:
        print(f"ERROR: Could not process video '{video_path}'. Reason: {e}")


def create_data_generators(dataset_dir: Path) -> Tuple[ImageDataGenerator, ImageDataGenerator]:
    """
    Creates and returns training and validation data generators.
    
    These generators handle data augmentation and preprocessing for the CNN model.
    They automatically split the data into training and validation sets.
    """
    datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,  # Normalize pixel values to [0,1]
        validation_split=VALIDATION_SPLIT
    )

    train_generator = datagen.flow_from_directory(
        dataset_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True,
    )

    val_generator = datagen.flow_from_directory(
        dataset_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=True,  # As in original script
    )
    return train_generator, val_generator


def create_cnn_model(input_shape: Tuple, num_classes: int) -> models.Sequential:
    """
    Builds and returns the CNN model architecture.
    
    This creates a standard CNN architecture suitable for video frame classification.
    The model consists of convolutional layers, pooling layers, and dense layers.
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])
    return model


def freeze_layers(model: models.Sequential, num_layers_to_freeze: int) -> models.Sequential:
    """
    Freezes the first `num_layers_to_freeze` layers of the model.
    
    This is a key technique in fine-tuning where early layers (which learn
    general features) are frozen while later layers are trained on new data.
    """
    for layer in model.layers[:num_layers_to_freeze]:
        layer.trainable = False
    print(f"INFO: First {num_layers_to_freeze} layers frozen for fine-tuning.")
    return model


def fine_tune_model(weights_path: Optional[Path], dataset_dir: Path) -> models.Sequential:
    """
    Orchestrates the model fine-tuning process, preserving the original logic.
    
    This function handles the complete fine-tuning workflow including model creation,
    weight loading, layer freezing, and training.
    """
    input_shape = (*IMG_SIZE, 3)
    
    # 1. Create model and load pre-trained weights if they exist
    model = create_cnn_model(input_shape, NUM_CLASSES)
    if weights_path and weights_path.exists():
        print(f"INFO: Loading pre-trained weights from: {weights_path}")
        model.load_weights(weights_path)
    else:
        print("INFO: No pre-trained weights specified or found. Training model from scratch.")

    # 2. Freeze earlier layers (EXACT LOGIC FROM ORIGINAL SCRIPT)
    if weights_path:
        num_layers_to_freeze = len(model.layers) - 6
        model = freeze_layers(model, num_layers_to_freeze)

    # 3. Recompile the model with a lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # 4. Prepare data generators
    train_gen, val_gen = create_data_generators(dataset_dir)

    # 5. Fine-tune the model
    print("\nStarting model fine-tuning...")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        verbose=1,
    )
    print("-> Fine-tuning complete.")
    return model


def main():
    """
    Main execution function to run the entire workflow.
    
    This function orchestrates the complete fine-tuning process:
    1. Configuration loading
    2. Frame extraction from videos
    3. Model fine-tuning
    4. Weight saving
    5. Parameter file updates
    """
    print("--- Starting Fine-Tuning Script ---")
    
    # --- Step 1: Handle Configuration and Paths ---
    try:
        video_targets = load_video_targets_from_config(CONFIG_FILE)
        pretrained_weights_path, new_weights_path = get_model_version_paths(MODEL_DIR, MODEL_WEIGHTS_BASENAME)
    except (FileNotFoundError, ValueError) as e:
        print(f"FATAL: A configuration error occurred. Exiting. Details: {e}")
        return

    # --- Step 2: Data Preparation (Frame Extraction) ---
    print("\n--- Phase 1: Extracting Frames from Videos ---")
    for item in video_targets:
        video_path = item.get("video_path")
        category = item.get("target_folder")
        if not video_path or not category:
            print(f"WARNING: Skipping invalid entry in config: {item}")
            continue
        
        target_dir = DATA_DIR / category
        extract_and_save_frames(video_path, target_dir, n_frames=50)

    # --- Step 3: Model Fine-Tuning ---
    print("\n--- Phase 2: Fine-Tuning the Model ---")
    if not any(DATA_DIR.iterdir()):
        print(f"ERROR: Data directory '{DATA_DIR}' is empty. No new frames were extracted. Aborting training.")
        return
        
    try:
        fine_tuned_model = fine_tune_model(
            weights_path=pretrained_weights_path,
            dataset_dir=DATA_DIR
        )
        
        # --- Step 4: Save the Newly Trained Model ---
        fine_tuned_model.save_weights(new_weights_path)
        print(f"\n✅ Successfully saved fine-tuned model weights to: {new_weights_path}")
        
        # --- Step 5: Update the main parameters file with the new model ID ---
        update_model_id_in_params(
            params_path=PARAMS_JSON_FILE,
            new_weights_path=new_weights_path,
            basename=MODEL_WEIGHTS_BASENAME
        )
        
    except Exception as e:
        print(f"\nFATAL: An unexpected error occurred during model training: {e}")

    print("\n--- Script Finished ---")


if __name__ == "__main__":
    main()

