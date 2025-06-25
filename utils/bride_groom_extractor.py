import numpy as np
import cv2
import os
import uuid

class BrideGroomExtractor:
    def __init__(self, yolo_model_base, yolo_model_face):
        """
        Initializes the BrideGroomExtractor with the YOLO models.

        :param yolo_model_base: The pre-trained YOLO model for detecting brides and grooms.
        :param yolo_model_face: The pre-trained YOLO model for detecting faces.
        """
        self.yolo_model_base = yolo_model_base
        self.yolo_model_face = yolo_model_face

    def _expand_bbox(self, x_min, y_min, x_max, y_max, image_shape, expand_ratio=1):
        """
        Expands the bounding box by a given ratio, ensuring it stays within image bounds.

        :param x_min: Minimum x-coordinate.
        :param y_min: Minimum y-coordinate.
        :param x_max: Maximum x-coordinate.
        :param y_max: Maximum y-coordinate.
        :param image_shape: Shape of the image as (height, width, channels).
        :param expand_ratio: Fraction by which to expand the box on each side.
        :return: Expanded bounding box coordinates (x_min, y_min, x_max, y_max).
        """
        height, width = image_shape[:2]
        box_width = x_max - x_min
        box_height = y_max - y_min

        x_min = max(0, int(x_min - expand_ratio * box_width))
        y_min = max(0, int(y_min - expand_ratio * box_height))
        x_max = min(width, int(x_max + expand_ratio * box_width))
        y_max = min(height, int(y_max + expand_ratio * box_height))

        return x_min, y_min, x_max, y_max

    def _contract_bbox(self, x_min, y_min, x_max, y_max, image_shape, contract_ratio=0.2, height_reduction_ratio=0.1):
        """
        Contracts the bounding box by a given ratio from the sides and reduces its height,
        ensuring it stays within image bounds.

        :param x_min: Minimum x-coordinate.
        :param y_min: Minimum y-coordinate.
        :param x_max: Maximum x-coordinate.
        :param y_max: Maximum y-coordinate.
        :param image_shape: Shape of the image as (height, width, channels).
        :param contract_ratio: Fraction by which to contract the box on each side (width-wise).
        :param height_reduction_ratio: Fraction by which to reduce the height of the box.
        :return: Contracted bounding box coordinates (x_min, y_min, x_max, y_max).
        """
        height, width = image_shape[:2]
        box_width = x_max - x_min
        box_height = y_max - y_min

        # Contract the width
        x_min = min(width, max(0, int(x_min + contract_ratio * box_width)))
        x_max = max(0, min(width, int(x_max - contract_ratio * box_width)))

        # Reduce the height
        height_reduction = int(height_reduction_ratio * box_height)
        y_min = min(height, max(0, y_min + height_reduction // 2))  # Reduce from the top
        y_max = max(0, min(height, y_max - height_reduction // 2))  # Reduce from the bottom

        return x_min, y_min, x_max, y_max

    def _detect_face(self, cropped_region):
        """
        Detects a face in the given cropped region using the face YOLO model.

        :param cropped_region: The cropped region of the image as a NumPy array.
        :return: Cropped face image and confidence score, or None if no face is detected.
        """
        # Ensure the input is in BGR format for consistency with the YOLO model
        if len(cropped_region.shape) == 3 and cropped_region.shape[2] == 3:  # RGB/BGR image
            cropped_region_bgr = cv2.cvtColor(cropped_region, cv2.COLOR_RGB2BGR)
        elif len(cropped_region.shape) == 2:  # Grayscale image
            cropped_region_bgr = cv2.cvtColor(cropped_region, cv2.COLOR_GRAY2BGR)
        else:  # Already BGR
            cropped_region_bgr = cropped_region


        ###### Optional #############################
        # # Save the image for debugging
        # os.makedirs("face_frames", exist_ok=True)
        # unique_filename = f"{uuid.uuid4()}.jpg"
        # cv2.imwrite(f"face_frames/{unique_filename}", cropped_region_bgr)
        ###### Optional #################################

        # Perform face detection using the YOLO model
        face_results = self.yolo_model_face.predict(cropped_region_bgr, conf=0.65, iou=0)
        face_boxes = face_results[0].boxes.xyxy.cpu().numpy()
        face_confidences = face_results[0].boxes.conf.cpu().numpy()

        if len(face_boxes) > 0:
            max_face_idx = np.argmax(face_confidences)
            x_min, y_min, x_max, y_max = map(int, face_boxes[max_face_idx])
            x_min, y_min, x_max, y_max = self._expand_bbox(x_min, y_min, x_max, y_max, cropped_region.shape)
            cropped_face = cropped_region[y_min:y_max, x_min:x_max]
            return cropped_face, face_confidences[max_face_idx]

        return None, None

    def _extract_person(self, image_array, indices, boxes, confidences):
        """
        Extracts a person (bride or groom) from the image based on the highest confidence score.

        :param image_array: The input image as a NumPy array.
        :param indices: Indices of detected persons (bride or groom).
        :param boxes: Bounding box coordinates.
        :param confidences: Confidence scores for detected persons.
        :return: Cropped face image and confidence score, or None if not found.
        """
        if len(indices) > 0:
            max_confidence_idx = indices[np.argmax(confidences[indices])]
            x_min, y_min, x_max, y_max = map(int, boxes[max_confidence_idx])
            # x_min, y_min, x_max, y_max = self._contract_bbox(x_min, y_min, x_max, y_max, image_array.shape)
            cropped_region = image_array[y_min:y_max, x_min:x_max]

            if cropped_region.size != 0:
                return self._detect_face(cropped_region)
        return None, None

    def extract_faces(self, image_array, confidence, card_name):
        """
        Extracts cropped bride and groom images with maximum confidence from the input image using the YOLO models.

        :param image_array: The input image as a NumPy array.
        :return: A dictionary with keys 'bride' and 'groom' containing cropped images of faces as NumPy arrays
                and their confidence scores.
        """
        results = self.yolo_model_base.predict(image_array, conf=confidence, iou=0)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()

        ############ OPTIONAL ##############################
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

        bride_indices = np.where(classes == 0)[0]  # index 0 is for the bride
        groom_indices = np.where(classes == 1)[0]  # index 1 is for the groom

        bride_and_groom_frame = None

        # Corrected condition
        if len(bride_indices) > 0 and len(groom_indices) > 0:
            print("yes")
            bride_and_groom_frame = image_array


        extracted_persons = {
            'bride': None,
            'groom': None
        }

        for person_type, indices in [('bride', bride_indices), ('groom', groom_indices)]:
            face_image, face_confidence = self._extract_person(image_array, indices, boxes, confidences)
            if face_image is not None:
                extracted_persons[person_type] = {
                    'image': face_image,
                    'confidence': face_confidence
                }

        return extracted_persons, bride_and_groom_frame










