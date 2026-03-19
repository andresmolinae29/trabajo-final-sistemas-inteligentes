import tempfile
import os

from pathlib import Path
import uuid
from basketball_detector.utils.logger import logger


BASE_DIR = Path(__file__).parent.parent.parent
TEMP_DIR = BASE_DIR / "static" / "tmp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)


class TempFileVideosManager:

    def __init__(self, ) -> None:
        self.temp_video_id = ""
        self.original_file = ""
        self.original_video_path = ""

    def set_names(self):
        self.temp_video_id = str(uuid.uuid4())
        self.original_file = self.temp_video_id + "_original.mp4"
        self.original_video_path = str(TEMP_DIR / self.original_file)
        
    def cleanup(self):

        if self.original_video_path and os.path.exists(self.original_video_path):
            try:
                os.remove(self.original_video_path)
            except FileNotFoundError | PermissionError | OSError as e:
                logger.warning(f"No se pudo eliminar el archivo temporal: {e}")
        
        else:
            logger.warning("No se encontró el archivo temporal para eliminar.")


if __name__ == "__main__":
    
    print(BASE_DIR)
