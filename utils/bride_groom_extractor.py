"""
DISCLAIMER - AMINDAV PROPERTY

This software is the exclusive property of Amindav. All rights reserved.
Unauthorized copying, distribution, or modification of this code is strictly prohibited.
This module is designed for bride and groom face extraction in wedding videos.

SYSTEM OVERVIEW:
This module provides advanced face extraction capabilities for the bride and groom
identification system. It uses YOLO models to:
1. Detect bride and groom persons in video frames
2. Extract and crop face regions from detected persons
3. Apply face detection to ensure quality face extraction
4. Handle bounding box expansion and contraction for optimal results
5. Provide confidence scores for extracted faces

WORKFLOW:
1. Receives video frames as input
2. Uses YOLO model to detect bride and groom persons
3. Crops person regions and applies face detection
4. Expands/contracts bounding boxes for optimal face extraction
5. Returns extracted faces with confidence scores

AUTHOR: Amindav Development Team
VERSION: 1.0
"""

import numpy as np
import cv2
import os
import uuid

class BrideGroomExtractor:
    """
    A class for extracting bride and groom faces from video frames using YOLO models.
    
    This class combines person detection and face detection to accurately extract
    bride and groom faces from wedding video frames. It uses two YOLO models:
    one for detecting bride/groom persons and another for face detection within
    the detected person regions.
    """
    
    def __init__(self, yolo_model_base, yolo_model_face):
        """
        Initializes the BrideGroomExtractor with the YOLO models.

        Args:
            yolo_model_base: The pre-trained YOLO model for detecting brides and grooms.
            yolo_model_face: The pre-trained YOLO model for detecting faces.
        """
        self.yolo_model_base = yolo_model_base
        self.yolo_model_face = yolo_model_face

    def _expand_bbox(self, x_min, y_min, x_max, y_max, image_shape, expand_ratio=1):
        """
        Expands the bounding box by a given ratio, ensuring it stays within image bounds.

        This method is used to expand person detection bounding boxes to capture
        more context around the person, which can improve face detection accuracy.

        Args:
            x_min: Minimum x-coordinate of the bounding box.
            y_min: Minimum y-coordinate of the bounding box.
            x_max: Maximum x-coordinate of the bounding box.
            y_max: Maximum y-coordinate of the bounding box.
            image_shape: Shape of the image as (height, width, channels).
            expand_ratio: Fraction by which to expand the box on each side.
            
        Returns:
            tuple: Expanded bounding box coordinates (x_min, y_min, x_max, y_max).
        """
        height, width = image_shape[:2]
        box_width = x_max - x_min
        box_height = y_max - y_min

        # Expand the bounding box while keeping it within image boundaries
        x_min = max(0, int(x_min - expand_ratio * box_width))
        y_min = max(0, int(y_min - expand_ratio * box_height))
        x_max = min(width, int(x_max + expand_ratio * box_width))
        y_max = min(height, int(y_max + expand_ratio * box_height))

        return x_min, y_min, x_max, y_max

    def _contract_bbox(self, x_min, y_min, x_max, y_max, image_shape, contract_ratio=0.2, height_reduction_ratio=0.1):
        """
        Contracts the bounding box by a given ratio from the sides and reduces its height,
        ensuring it stays within image bounds.

        This method is used to focus the detection area more tightly around the person,
        which can improve face detection precision by reducing background noise.

        Args:
            x_min: Minimum x-coordinate of the bounding box.
            y_min: Minimum y-coordinate of the bounding box.
            x_max: Maximum x-coordinate of the bounding box.
            y_max: Maximum y-coordinate of the bounding box.
            image_shape: Shape of the image as (height, width, channels).
            contract_ratio: Fraction by which to contract the box on each side (width-wise).
            height_reduction_ratio: Fraction by which to reduce the height of the box.
            
        Returns:
            tuple: Contracted bounding box coordinates (x_min, y_min, x_max, y_max).
        """
        height, width = image_shape[:2]
        box_width = x_max - x_min
        box_height = y_max - y_min

        # Contract the width from both sides
        x_min = min(width, max(0, int(x_min + contract_ratio * box_width)))
        x_max = max(0, min(width, int(x_max - contract_ratio * box_width)))

        # Reduce the height from top and bottom
        height_reduction = int(height_reduction_ratio * box_height)
        y_min = min(height, max(0, y_min + height_reduction // 2))  # Reduce from the top
        y_max = max(0, min(height, y_max - height_reduction // 2))  # Reduce from the bottom

        return x_min, y_min, x_max, y_max

    def _detect_face(self, cropped_region):
        """
        Detects a face in the given cropped region using the face YOLO model.

        This method applies face detection to a cropped person region to ensure
        that we extract actual face images rather than just person regions.

        Args:
            cropped_region: The cropped region of the image as a NumPy array.
            
        Returns:
            tuple: (cropped_face_image, confidence_score) or (None, None) if no face detected.
        """
        # Ensure the input is in BGR format for consistency with the YOLO model
        if len(cropped_region.shape) == 3 and cropped_region.shape[2] == 3:  # RGB/BGR image
            cropped_region_bgr = cv2.cvtColor(cropped_region, cv2.COLOR_RGB2BGR)
        elif len(cropped_region.shape) == 2:  # Grayscale image
            cropped_region_bgr = cv2.cvtColor(cropped_region, cv2.COLOR_GRAY2BGR)
        else:  # Already BGR
            cropped_region_bgr = cropped_region

        ###### Optional Debugging Code #############################
        # # Save the image for debugging
        # os.makedirs("face_frames", exist_ok=True)
        # unique_filename = f"{uuid.uuid4()}.jpg"
        # cv2.imwrite(f"face_frames/{unique_filename}", cropped_region_bgr)
        ###### Optional Debugging Code #################################

        # Perform face detection using the YOLO model with confidence threshold
        face_results = self.yolo_model_face.predict(cropped_region_bgr, conf=0.65, iou=0)
        face_boxes = face_results[0].boxes.xyxy.cpu().numpy()
        face_confidences = face_results[0].boxes.conf.cpu().numpy()

        if len(face_boxes) > 0:
            # Select the face with the highest confidence
            max_face_idx = np.argmax(face_confidences)
            x_min, y_min, x_max, y_max = map(int, face_boxes[max_face_idx])
            # Expand the face bounding box to capture more context
            x_min, y_min, x_max, y_max = self._expand_bbox(x_min, y_min, x_max, y_max, cropped_region.shape)
            cropped_face = cropped_region[y_min:y_max, x_min:x_max]
            return cropped_face, face_confidences[max_face_idx]

        return None, None

    def _extract_person(self, image_array, indices, boxes, confidences):
        """
        Extracts a person (bride or groom) from the image based on the highest confidence score.

        This method processes detected persons and extracts their face regions.
        It selects the person with the highest confidence score and applies
        face detection to ensure quality face extraction.

        Args:
            image_array: The input image as a NumPy array.
            indices: Indices of detected persons (bride or groom).
            boxes: Bounding box coordinates for all detections.
            confidences: Confidence scores for all detections.
            
        Returns:
            tuple: (cropped_face_image, confidence_score) or (None, None) if not found.
        """
        if len(indices) > 0:
            # Select the person with the highest confidence
            max_confidence_idx = indices[np.argmax(confidences[indices])]
            x_min, y_min, x_max, y_max = map(int, boxes[max_confidence_idx])
            # Note: Contracting bounding box is commented out but available if needed
            # x_min, y_min, x_max, y_max = self._contract_bbox(x_min, y_min, x_max, y_max, image_array.shape)
            cropped_region = image_array[y_min:y_max, x_min:x_max]

            if cropped_region.size != 0:
                return self._detect_face(cropped_region)
        return None, None

    def extract_faces(self, image_array, confidence, card_name):
        """
        Extracts cropped bride and groom images with maximum confidence from the input image.

        This is the main method that orchestrates the face extraction process.
        It detects bride and groom persons, extracts their faces, and returns
        the results along with a frame that contains both bride and groom.

        Args:
            image_array: The input image as a NumPy array.
            confidence: Confidence threshold for person detection.
            card_name: Name of the project card for debugging purposes.
            
        Returns:
            tuple: (extracted_persons_dict, bride_and_groom_frame)
                - extracted_persons_dict: Dictionary with 'bride' and 'groom' keys containing
                  face images and confidence scores
                - bride_and_groom_frame: The original frame if both bride and groom are detected
        """
        # Perform person detection using the base YOLO model
        results = self.yolo_model_base.predict(image_array, conf=confidence, iou=0)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()

        ############ Optional Debugging Code ##############################
        # # Save the original image with bounding boxes and confidence scores
        # os.makedirs(f"{card_name}/complete_frames", exist_ok=True)
        # os.makedirs(f"{card_name}/cut_frames", exist_ok=True)
        # for i, box in enumerate(boxes):
        #     x_min, y_min, x_max, y_max = map(int, box)
        #     confidence_score = confidences[i]
        #     label = f"{confidence_score:.2f}"

        #     cropped_region = image_array[y_min:y_max, x_min:x_max]

        #     # Generate a unique filename using uuid
        #     unique_filename = f"{uuid.uuid4()}.jpg"

        #     # Save the image with all bounding boxes
        #     cv2.imwrite(f"{card_name}/cut_frames/{unique_filename}", cv2.cvtColor(cropped_region, cv2.COLOR_RGB2BGR))

        # for i, box in enumerate(boxes):
        #     x_min, y_min, x_max, y_max = map(int, box)
        #     confidence_score = confidences[i]
        #     label = f"{confidence_score:.2f}"

        #     # Draw the bounding box on the image
        #     cv2.rectangle(image_array, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        #     cv2.putText(image_array, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # # Generate a unique filename using uuid
        # unique_filename = f"{uuid.uuid4()}.jpg"

        # # Save the image with all bounding boxes
        # cv2.imwrite(f"{card_name}/complete_frames/{unique_filename}", cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
        #############################################

        # Separate bride and groom detections based on class indices
        bride_indices = np.where(classes == 0)[0]  # index 0 is for the bride
        groom_indices = np.where(classes == 1)[0]  # index 1 is for the groom

        bride_and_groom_frame = None

        # Check if both bride and groom are detected in the same frame
        if len(bride_indices) > 0 and len(groom_indices) > 0:
            print("yes")
            bride_and_groom_frame = image_array

        # Initialize result dictionary
        extracted_persons = {
            'bride': None,
            'groom': None
        }

        # Extract faces for both bride and groom
        for person_type, indices in [('bride', bride_indices), ('groom', groom_indices)]:
            face_image, face_confidence = self._extract_person(image_array, indices, boxes, confidences)
            if face_image is not None:
                extracted_persons[person_type] = {
                    'image': face_image,
                    'confidence': face_confidence
                }

        return extracted_persons, bride_and_groom_frame










