import tempfile
import os

from .logger import logger


class TempFileVideosManager:

    def __init__(self, ) -> None:
        
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        self.temp_video_path = self.temp_file.name
        self.temp_file.close()
        
    def cleanup(self):
        try:
            os.remove(self.temp_video_path)
        except FileNotFoundError | PermissionError | OSError as e:
            logger.warning(f"⚠️ No se pudo eliminar el archivo temporal: {e}")

if __name__ == "__main__":
    pass
