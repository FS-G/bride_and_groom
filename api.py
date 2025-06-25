"""
DISCLAIMER - AMINDAV PROPERTY

This software is the exclusive property of Amindav. All rights reserved.
Unauthorized copying, distribution, or modification of this code is strictly prohibited.
This API service is designed for face verification and grouping in the bride and groom identification system.

SYSTEM OVERVIEW:
This is the parallel API service for face processing using DeepFace.
The API receives face images from the main application and:
1. Groups similar faces using DeepFace verification
2. Returns the best quality face group for each person
3. Uses adaptive thresholding to ensure optimal grouping
4. Provides face verification services to the main application

WORKFLOW:
1. Receives face images via HTTP POST request
2. Saves images to temporary directory
3. Uses DeepFace to verify and group similar faces
4. Returns grouped face paths to the main application
5. Cleans up temporary files

AUTHOR: Amindav Development Team
VERSION: 1.0
"""

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from typing import List
from deepface import DeepFace
import numpy as np
import cv2
from io import BytesIO
import os
import shutil
import uuid

# Initialize FastAPI application for face processing service
app = FastAPI()


@app.post("/process_face/")
async def process_faces(files: List[UploadFile] = File(...), model_name: str = "Facenet512", detector_backend: str = "retinaface", distance_threshold: float = 0.35):
    """
    Processes a list of face images to group similar faces using DeepFace.
    
    This endpoint receives face images from the main application and uses DeepFace
    to verify and group similar faces. It implements adaptive thresholding to
    ensure optimal grouping even when faces are challenging to match.
    
    Args:
        files (List[UploadFile]): List of uploaded face images
        model_name (str): DeepFace model to use for face verification (default: "Facenet512")
        detector_backend (str): Face detection backend (default: "retinaface")
        distance_threshold (float): Initial distance threshold for face matching (default: 0.35)
        
    Returns:
        JSONResponse: Contains grouped face paths or error information
        
    Raises:
        Exception: If any error occurs during processing
    """
    print('Request received')
    try:
        # Create a temporary directory to store uploaded images
        # This ensures clean processing and prevents file conflicts
        temp_dir = "temp_images"

        # Remove existing directory and create a fresh one
        # This prevents accumulation of old files and ensures clean state
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)

        image_paths = []

        # Save all uploaded images to the temp directory with unique names
        # Unique names prevent conflicts when multiple requests are processed
        for file in files:
            print("Writing data ..")
            unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
            image_path = os.path.join(temp_dir, unique_filename)
            with open(image_path, "wb") as f:
                f.write(await file.read())
            image_paths.append(image_path)

        # Keep a copy of original image paths for multiple iteration attempts
        image_paths_original = image_paths.copy()

        # Process images to group similar faces using adaptive thresholding
        # Start with strict threshold and gradually relax if needed
        threshold_step = 0.1  # Increment step for threshold relaxation
        max_iterations = 5  # Safety cap to prevent infinite loops
        iteration_count = 0
        current_threshold = distance_threshold

        while True:
            groups = []
            
            # Make a fresh copy of original paths in each iteration
            # This ensures we start with all images in each attempt
            image_paths = image_paths_original.copy()
            is_first_iteration = iteration_count == 0  # Flag to identify the first iteration

            while image_paths:
                # Select base image for comparison
                base_image_path = image_paths.pop(0)
                current_group = [base_image_path]
                images_to_remove = []
                print("Base image:", base_image_path)

                # Compare base image with all remaining images
                for other_image_path in image_paths:
                    try:
                        # Use DeepFace to verify if two faces belong to the same person
                        result = DeepFace.verify(
                            img1_path=base_image_path,
                            img2_path=other_image_path,
                            model_name=model_name,
                            detector_backend=detector_backend,
                            enforce_detection=False
                        )
                        
                        # First iteration: check both "verified" and "distance"
                        # This ensures high confidence matches in the first pass
                        if is_first_iteration:
                            if result["verified"] and result["distance"] <= current_threshold:
                                print("Verified as:", other_image_path, "Distance:", result["distance"])
                                current_group.append(other_image_path)
                                images_to_remove.append(other_image_path)
                        # Subsequent iterations: check only "distance"
                        # This allows for more relaxed matching in later attempts
                        else:
                            if result["distance"] <= current_threshold:
                                print("Relaxed verification:", other_image_path, "Distance:", result["distance"])
                                current_group.append(other_image_path)
                                images_to_remove.append(other_image_path)

                        # Return if the group reaches 5 images (optimal group size)
                        if len(current_group) == 5:
                            print('Group of 5 found.')
                            return JSONResponse(content={"group": current_group}, status_code=200)
                    except Exception as e:
                        print(f"Error comparing images: {e}")

                # Remove matched images from the list to prevent reprocessing
                # image_paths = [img for img in image_paths if img not in images_to_remove]
                groups.append(current_group)

            # Increment threshold for subsequent iterations
            # This allows for more relaxed matching if strict matching fails
            iteration_count += 1
            if iteration_count >= max_iterations:
                print("Max iterations reached, stopping.")
                break
            current_threshold += threshold_step
            print(f"Increasing threshold to {current_threshold}")
            

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)



