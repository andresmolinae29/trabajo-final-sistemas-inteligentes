import cv2
import numpy as np

from typing import List


class VideoWriter:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def write_video(self, frames: List[np.ndarray], fps: float = 4.0, height: int = 0, width: int = 0):
        if not frames or not isinstance(frames, list) or len(frames) == 0:
            raise ValueError("No hay frames válidos para procesar")

        if height or width:
            height, width = frames[0].shape[:2]

        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        out = cv2.VideoWriter(self.file_path, fourcc, fps, (width, height))

        for frame in frames:
            out.write(frame)

        out.release()