# import os
# from pymediainfo import MediaInfo

# class VideoFinder:
#     """
#     A utility class to locate videos based on a directory structure, 
#     card name, and a file-length condition.
#     """

#     def __init__(self, directories, video_extensions=None):
#         """
#         Initializes the VideoFinder.

#         Args:
#             directories (list): A list of base directories to search.
#             video_extensions (tuple): Supported video file extensions.
#         """
#         self.directories = directories
#         self.video_extensions = video_extensions or ('.mp4', '.avi', '.mkv', '.mov', '.flv')

#     def find_matching_directory(self, target_name):
#         """
#         Searches for a directory matching the target name within the base directories.

#         Args:
#             target_name (str): The name of the directory to search for.

#         Returns:
#             str: Path of the matching directory, or None if not found.
#         """
#         for base_dir in self.directories:
#             if not os.path.exists(base_dir):
#                 continue

#             for root, subdirs, _ in os.walk(base_dir):
#                 if target_name in subdirs:
#                     return os.path.join(root, target_name)
        
#         print(f"No matching directory found with the name '{target_name}'.")
#         return "NULL"

#     def get_video_durations(self, directory, file_length):
#         """
#         Collects videos from a directory that exceed a specified length.

#         Args:
#             directory (str): The directory to search for videos.
#             file_length (int): Minimum length of videos in seconds.

#         Returns:
#             list: Sorted list of video paths exceeding the file length.
#         """
#         video_durations = {}
        
#         for dirpath, _, filenames in os.walk(directory):
#             for filename in filenames:
#                 if filename.lower().endswith(self.video_extensions):
#                     file_path = os.path.join(dirpath, filename)
#                     duration = self._get_video_duration(file_path)
#                     if duration and duration > file_length:
#                         video_durations[file_path] = duration
        
#         return sorted(video_durations, key=video_durations.get, reverse=True)

#     def _get_video_duration(self, file_path):
#         """
#         Extracts the duration of a video using pymediainfo.

#         Args:
#             file_path (str): Path to the video file.

#         Returns:
#             float: Duration of the video in seconds, or None if not found.
#         """
#         try:
#             media_info = MediaInfo.parse(file_path)
#             for track in media_info.tracks:
#                 if track.track_type == "Video":
#                     return track.duration / 1000  # Convert to seconds
#         except Exception as e:
#             print(f"Error processing video {file_path}: {e}")
#         return None

#     def find_videos(self, card_name, file_length):
#         """
#         Finds videos in a matching directory that exceed a specified length.

#         Args:
#             card_name (str): The target card name to locate the directory.
#             file_length (int): Minimum length of videos in seconds.

#         Returns:
#             list: Sorted list of video file paths exceeding the file length.
#         """
#         matching_dir = self.find_matching_directory(card_name)
#         if matching_dir == "NULL":
#             return "NULL"
#         elif not matching_dir:
#             return []

#         return self.get_video_durations(matching_dir, file_length)
    

















# # import os
# # from pymediainfo import MediaInfo

# # class VideoFinder:
# #     """
# #     A utility class to locate videos based on a directory structure, 
# #     card name, and a file-length condition.
# #     """

# #     def __init__(self, directories, video_extensions=None):
# #         """
# #         Initializes the VideoFinder.

# #         Args:
# #             directories (list): A list of base directories to search.
# #             video_extensions (tuple): Supported video file extensions.
# #         """
# #         self.directories = directories
# #         self.video_extensions = video_extensions or ('.mp4', '.avi', '.mkv', '.mov', '.flv')

# #     def find_matching_directory(self, target_name):
# #         """
# #         Searches for a directory matching the target name within the base directories.

# #         Args:
# #             target_name (str): The name of the directory to search for.

# #         Returns:
# #             str: Path of the matching directory, or None if not found.
# #         """
# #         for base_dir in self.directories:
# #             if not os.path.exists(base_dir):
# #                 continue

# #             for root, subdirs, _ in os.walk(base_dir):
# #                 if target_name in subdirs:
# #                     return os.path.join(root, target_name)
        
# #         print(f"No matching directory found with the name '{target_name}'.")
# #         return None

# #     def get_video_durations(self, directory, file_length):
# #         """
# #         Collects videos from a directory that exceed a specified length.

# #         Args:
# #             directory (str): The directory to search for videos.
# #             file_length (int): Minimum length of videos in seconds.

# #         Returns:
# #             list: Sorted list of video paths exceeding the file length.
# #         """
# #         video_durations = {}
        
# #         for dirpath, _, filenames in os.walk(directory):
# #             for filename in filenames:
# #                 if filename.lower().endswith(self.video_extensions):
# #                     file_path = os.path.join(dirpath, filename)
# #                     duration = self._get_video_duration(file_path)
# #                     if duration and duration > file_length:
# #                         video_durations[file_path] = duration
        
# #         return sorted(video_durations, key=video_durations.get, reverse=True)

# #     def _get_video_duration(self, file_path):
# #         """
# #         Extracts the duration of a video using pymediainfo.

# #         Args:
# #             file_path (str): Path to the video file.

# #         Returns:
# #             float: Duration of the video in seconds, or None if not found.
# #         """
# #         try:
# #             media_info = MediaInfo.parse(file_path)
# #             for track in media_info.tracks:
# #                 if track.track_type == "Video":
# #                     return track.duration / 1000  # Convert to seconds
# #         except Exception as e:
# #             print(f"Error processing video {file_path}: {e}")
# #         return None

# #     def find_videos(self, card_name, file_length):
# #         """
# #         Finds videos in a matching directory that exceed a specified length.

# #         Args:
# #             card_name (str): The target card name to locate the directory.
# #             file_length (int): Minimum length of videos in seconds.

# #         Returns:
# #             list: Sorted list of video file paths exceeding the file length.
# #         """
# #         matching_dir = self.find_matching_directory(card_name)
# #         if not matching_dir:
# #             return []

# #         return self.get_video_durations(matching_dir, file_length)
    




import os
from pymediainfo import MediaInfo

class VideoFinder:
    """
    A utility class to locate videos based on a directory structure, 
    card name, and a file-length condition.
    """

    def __init__(self, directories, video_extensions=None):
        """
        Initializes the VideoFinder.

        Args:
            directories (list): A list of base directories to search.
            video_extensions (tuple): Supported video file extensions.
        """
        self.directories = directories
        self.video_extensions = video_extensions or ('.mp4', '.avi', '.mkv', '.mov', '.flv')

    def find_matching_directory(self, target_name):
        """
        Searches for a directory matching the target name within the base directories.

        Args:
            target_name (str): The name of the directory to search for.

        Returns:
            str: Path of the matching directory, or None if not found.
        """
        for base_dir in self.directories:
            if not os.path.exists(base_dir):
                continue

            for root, subdirs, _ in os.walk(base_dir):
                if target_name in subdirs:
                    return os.path.join(root, target_name)
        
        print(f"No matching directory found with the name '{target_name}'.")
        return "NULL"

    def get_video_durations(self, directory, file_length):
        """
        Collects videos from a directory that exceed a specified length.

        Args:
            directory (str): The directory to search for videos.
            file_length (int): Minimum length of videos in seconds.

        Returns:
            list: Sorted list of video paths exceeding the file length.
        """
        video_durations = {}
        
        for dirpath, _, filenames in os.walk(directory):
            # Skip directories with "error" in their path
            if "error" in os.path.normpath(dirpath).split(os.sep):
                continue

            for filename in filenames:
                if filename.lower().endswith(self.video_extensions):
                    file_path = os.path.join(dirpath, filename)
                    duration = self._get_video_duration(file_path)
                    if duration and duration > file_length:
                        video_durations[file_path] = duration
        
        return sorted(video_durations, key=video_durations.get, reverse=True)

    def _get_video_duration(self, file_path):
        """
        Extracts the duration of a video using pymediainfo.

        Args:
            file_path (str): Path to the video file.

        Returns:
            float: Duration of the video in seconds, or None if not found.
        """
        try:
            media_info = MediaInfo.parse(file_path)
            for track in media_info.tracks:
                if track.track_type == "Video":
                    return track.duration / 1000  # Convert to seconds
        except Exception as e:
            print(f"Error processing video {file_path}: {e}")
        return None

    def find_videos(self, card_name, file_length):
        """
        Finds videos in a matching directory that exceed a specified length.

        Args:
            card_name (str): The target card name to locate the directory.
            file_length (int): Minimum length of videos in seconds.

        Returns:
            list: Sorted list of video file paths exceeding the file length.
        """
        matching_dir = self.find_matching_directory(card_name)
        if matching_dir == "NULL":
            return "NULL"
        elif not matching_dir:
            return []

        return self.get_video_durations(matching_dir, file_length)





