import cv2
import numpy as np

from basketball_detector.services.detection_service import DetectionService
from basketball_detector.ai_models.models import VideoModel, Qwen2VideoModel, GoogleVideoModel


class VideoService:

    def __init__(self, detection_service: DetectionService, capture_frames: int = 60, video_model: VideoModel = Qwen2VideoModel()):
        self.detection_service = detection_service
        self.capture_frames = capture_frames  # Cuántos frames capturar (por defecto 20 = ~3s a 30fps)
        self.video_model = video_model

    def __enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        """Mejora contraste y saturación antes de enviar a Gemini"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], 1.5)  
        hsv[:, :, 2] = cv2.multiply(hsv[:, :, 2], 1.2)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def process_video(self, video_path: str | int = 0):
        """
        Procesa video capturando tomas de basquetbol.
        
        Args:
            video_path: Ruta del video o ID de cámara
        """
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_BRIGHTNESS, 30)
        cap.set(cv2.CAP_PROP_CONTRAST, 40)
        cap.set(cv2.CAP_PROP_SATURATION, 60)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        decisions = []
        capturing = False
        captured_frames = []
        shot_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if not capturing:
                detections = self.detection_service.detect_objects(frame)
                if self.detection_service.balled_ball_in_frame(detections):
                    capturing = True
                    captured_frames = []

            else:
                captured_frames.append(self.__enhance_frame(frame.copy()))

                # Si completamos los frames requeridos
                if len(captured_frames) >= self.capture_frames:
                    shot_count += 1
                    decision = self.video_model.predict(captured_frames)
                    decisions.append(decision)
                    
                    capturing = False
                    captured_frames = []

            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        return decisions


if __name__ == "__main__":
    
    video_service = VideoService(DetectionService(), video_model=GoogleVideoModel())
    video_service.process_video(
        r"C:\dev\trabajo-final-sistemas-inteligentes\tests\test_videos\2-video.mp4",
    )
