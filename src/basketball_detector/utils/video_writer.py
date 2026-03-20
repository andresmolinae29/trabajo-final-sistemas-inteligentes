import cv2
import numpy as np
import subprocess

from typing import List
from basketball_detector.utils.manage_temp_files import TempFileVideosManager, BASE_DIR


class VideoWriter:
    def __init__(self, file_path: str | None = None):
        self.file_path = file_path

    def write_video(self, frames: List[np.ndarray], fps: float = 4.0, height: int = 0, width: int = 0):
        if not frames or not isinstance(frames, list) or len(frames) == 0:
            raise ValueError("No hay frames válidos para procesar")

        if height or width:
            height, width = frames[0].shape[:2]

        fourcc = cv2.VideoWriter.fourcc(*'mp4v')

        if not self.file_path:
            raise ValueError("No se ha especificado un file_path para guardar el video")

        out = cv2.VideoWriter(self.file_path, fourcc, fps, (width, height))

        for frame in frames:
            out.write(frame)

        out.release()

    @staticmethod
    def fix_mp4_faststart(video_path: str) -> str:
        if video_path and video_path.endswith('.mp4'):
            fixed_path = video_path.replace('.mp4', '_faststart.mp4')
            cmd = [
                'ffmpeg', '-y', '-i', f"{BASE_DIR}/{video_path}",
                '-movflags', 'faststart', f"{BASE_DIR}/{fixed_path}"
            ]
            try:
                subprocess.run(cmd, check=True)
                return fixed_path
            except Exception as e:
                print(f'Error al convertir video para faststart: {e}')
        return video_path