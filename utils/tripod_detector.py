import cv2
import numpy as np
from moviepy.editor import VideoFileClip
import os

class TripodDetector:
    def __init__(self, motion_threshold=1.0, feature_params=None, lk_params=None):
        """
        Initialize the TripodDetector class.
        
        Args:
            motion_threshold (float): Threshold for global camera motion.
            feature_params (dict): Parameters for cv2.goodFeaturesToTrack.
            lk_params (dict): Parameters for cv2.calcOpticalFlowPyrLK.
        """
        self.motion_threshold = motion_threshold
        self.feature_params = feature_params or dict(
            maxCorners=200,
            qualityLevel=0.01,
            minDistance=30,
            blockSize=3
        )
        self.lk_params = lk_params or dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

    def _detect_camera_movement(self, video_path):
        """
        Analyze global camera movement in a video by focusing on the background motion.
        
        Args:
            video_path (str): Path to the video file.
            
        Returns:
            str: "Still Camera" if background motion is below threshold, else "Moving Camera".
        """
        # Load the video using MoviePy for frame extraction
        clip = VideoFileClip(video_path)
        duration = clip.duration
        
        # Create a list of timestamps to sample frames (e.g., every 10 seconds, up to 20 frames)
        frame_intervals = [i * 10 for i in range(20) if i * 10 < duration]
        
        prev_gray = None
        camera_motion_magnitudes = []

        for t in frame_intervals:
            # Extract the frame at the given timestamp
            frame = clip.get_frame(t)
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            if prev_gray is not None:
                # Detect good features to track in the previous frame
                p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **self.feature_params)
                if p0 is None or len(p0) < 4:
                    prev_gray = gray
                    continue

                # Calculate optical flow using the Lucas-Kanade method
                p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **self.lk_params)
                if p1 is None or st is None:
                    prev_gray = gray
                    continue
                
                # Filter out the valid points
                good_old = p0[st.flatten() == 1]
                good_new = p1[st.flatten() == 1]
                
                if len(good_old) < 4 or len(good_new) < 4:
                    prev_gray = gray
                    continue

                # Estimate a partial affine transformation (translation, rotation, scale) between frames.
                M, inliers = cv2.estimateAffinePartial2D(good_old, good_new, method=cv2.RANSAC)
                if M is None:
                    prev_gray = gray
                    continue
                
                # Extract the translation components from the transformation matrix
                dx = M[0, 2]
                dy = M[1, 2]
                translation_magnitude = np.sqrt(dx**2 + dy**2)
                camera_motion_magnitudes.append(translation_magnitude)
            
            prev_gray = gray

        clip.close()

        # Calculate average camera motion over the sampled frames
        average_motion = np.mean(camera_motion_magnitudes) if camera_motion_magnitudes else 0
        print(f"Video path: {video_path}, Average Camera Motion: {average_motion}")

        # Classify based on the provided motion threshold
        if average_motion < self.motion_threshold:
            return "Still Camera"
        else:
            return "Moving Camera"

    def find_tripod_videos(self, video_paths):
        """
        Identify videos with minimal camera motion ("Still Camera").
        
        Args:
            video_paths (list of str): List of video file paths.
            
        Returns:
            list of str: List of video paths classified as "Still Camera".
        """
        tripod_videos = []
        folder_to_consider = None

        for video_path in video_paths:
            # If a folder is already selected, only process files from that folder
            if folder_to_consider and not video_path.startswith(folder_to_consider):
                continue

            camera_type = self._detect_camera_movement(video_path)
            if camera_type == "Still Camera":
                tripod_videos.append(video_path)
                # Optionally limit to one folder for processing
                if not folder_to_consider:
                    folder_to_consider = os.path.dirname(video_path) + os.sep

        return tripod_videos