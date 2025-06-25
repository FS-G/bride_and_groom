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

app = FastAPI()





@app.post("/process_face/")
async def process_faces(files: List[UploadFile] = File(...), model_name: str = "Facenet512", detector_backend: str = "retinaface", distance_threshold: float = 0.35):
    print('Request received')
    try:
        # Create a temporary directory to store uploaded images
        temp_dir = "temp_images"

        # Remove existing directory and create a fresh one
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)

        image_paths = []

        # Save all uploaded images to the temp directory with unique names
        for file in files:
            print("Writing data ..")
            unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
            image_path = os.path.join(temp_dir, unique_filename)
            with open(image_path, "wb") as f:
                f.write(await file.read())
            image_paths.append(image_path)

        # Keep a copy of original image paths
        image_paths_original = image_paths.copy()

        # Process images to group similar faces
        threshold_step = 0.1  # Increment step for threshold
        max_iterations = 5  # Safety cap to prevent infinite loops
        iteration_count = 0
        current_threshold = distance_threshold

        while True:
            groups = []
            

            # Make a fresh copy of original paths in each iteration
            image_paths = image_paths_original.copy()
            is_first_iteration = iteration_count == 0  # Flag to identify the first iteration

            while image_paths:
                base_image_path = image_paths.pop(0)
                current_group = [base_image_path]
                images_to_remove = []
                print("Base image:", base_image_path)

                # Compare base image with others
                for other_image_path in image_paths:
                    try:
                        result = DeepFace.verify(
                            img1_path=base_image_path,
                            img2_path=other_image_path,
                            model_name=model_name,
                            detector_backend=detector_backend,
                            enforce_detection=False
                        )
                        # First iteration: check both "verified" and "distance"
                        if is_first_iteration:
                            if result["verified"] and result["distance"] <= current_threshold:
                                print("Verified as:", other_image_path, "Distance:", result["distance"])
                                current_group.append(other_image_path)
                                images_to_remove.append(other_image_path)
                        # Subsequent iterations: check only "distance"
                        else:
                            if result["distance"] <= current_threshold:
                                print("Relaxed verification:", other_image_path, "Distance:", result["distance"])
                                current_group.append(other_image_path)
                                images_to_remove.append(other_image_path)

                        # Return if the group reaches 5 images
                        if len(current_group) == 5:
                            print('Group of 5 found.')
                            return JSONResponse(content={"group": current_group}, status_code=200)
                    except Exception as e:
                        print(f"Error comparing images: {e}")

                # Remove matched images from the list
                # image_paths = [img for img in image_paths if img not in images_to_remove]
                groups.append(current_group)

            # Increment threshold for subsequent iterations
            iteration_count += 1
            if iteration_count >= max_iterations:
                print("Max iterations reached, stopping.")
                break
            current_threshold += threshold_step
            print(f"Increasing threshold to {current_threshold}")
            

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)



