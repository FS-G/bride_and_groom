import cv2
import numpy as np
from moviepy.editor import VideoFileClip

class TripodDetector:
    def __init__(self, motion_threshold=1.0):
        """
        Initialize the TripodDetector class.

        Args:
            motion_threshold (float): Threshold for motion to classify as a moving camera.
        """
        self.motion_threshold = motion_threshold

    def _detect_camera_movement(self, video_path):
        """
        (Internal) Classifies the type of camera movement in a video using MoviePy for frame extraction.

        Args:
            video_path (str): Path to the video file.

        Returns:
            str: "Still Camera" or "Moving Camera".
        """
        # Load the video using MoviePy
        clip = VideoFileClip(video_path)
        duration = clip.duration  # Get the duration of the video

        # Calculate 20 evenly spaced timestamps, spaced at 10 seconds or less if the video is short
        frame_intervals = [i * 20 for i in range(80) if i * 20 *80 < duration]

        prev_gray = None
        motion_magnitudes = []

        for t in frame_intervals:
            # Extract the frame at the given timestamp
            frame = clip.get_frame(t)
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # Convert to grayscale

            # Calculate optical flow if we have a previous frame
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                                    pyr_scale=0.5, levels=3, winsize=15,
                                                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
                magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                motion_magnitudes.append(np.mean(magnitude))

            prev_gray = gray

        # Close the video file
        clip.close()

        # Analyze motion magnitudes
        average_motion = np.mean(motion_magnitudes) if motion_magnitudes else 0
        print(average_motion)
        print(f"video path: {video_path} , Average motion:{average_motion}")

        # Classify based on motion threshold
        if average_motion < self.motion_threshold:
            return "Still Camera"
        else:
            return "Moving Camera"

    def find_tripod_videos(self, video_paths):
        """
        Identifies videos with "Still Camera" classification.

        Args:
            video_paths (list of str): List of video file paths.

        Returns:
            list of str: List of video paths classified as "Still Camera".
        """
        tripod_videos = []
        for video_path in video_paths:
            camera_type = self._detect_camera_movement(video_path)
            if camera_type == "Still Camera":
                tripod_videos.append(video_path)
        return tripod_videos
