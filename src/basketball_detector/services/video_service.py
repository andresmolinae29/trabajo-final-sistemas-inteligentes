import cv2
import numpy as np

from basketball_detector.ai_models import ModelWrapperBase, GoogleVideoModelFactory
from basketball_detector.services import BasketballDetectionService, DetectionServiceFactory


class VideoService:

    def __init__(self, detection_service: BasketballDetectionService, video_model: ModelWrapperBase, capture_frames: int = 60):
        self.detection_service = detection_service
        self.capture_frames = capture_frames
        self.video_model = video_model

    def __enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        """Mejora contraste y saturación antes de enviar a Gemini"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], 1.5) # type: ignore
        hsv[:, :, 2] = cv2.multiply(hsv[:, :, 2], 1.2) # type: ignore
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
                if detections:
                    capturing = True
                    captured_frames = []

            else:
                captured_frames.append(self.__enhance_frame(frame.copy()))

                # Si completamos los frames requeridos
                if len(captured_frames) >= self.capture_frames:
                    shot_count += 1
                    decision = self.video_model.llm_predict(captured_frames)
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
    
    basketball_service = DetectionServiceFactory.create()
    google_video_model = GoogleVideoModelFactory.create()

    video_service = VideoService(basketball_service, google_video_model)
    decisions = video_service.process_video(
        r"C:\dev\trabajo-final-sistemas-inteligentes\tests\test_videos\2-video.mp4",
    )

    print(decisions)