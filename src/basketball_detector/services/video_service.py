import cv2
import numpy as np
from basketball_detector.ai_models import ModelWrapperBase, GoogleVideoModelFactory
from basketball_detector.services import (
    BasketballDetectionService,
    DetectionServiceFactory,
)
from basketball_detector.models.detection_model import VideoResponse


class CameraConfig:
    def __init__(
        self,
        width=1280,
        height=720,
        brightness=30,
        contrast=40,
        saturation=60,
        autofocus=1,
        auto_exposure=1,
        fps=30,
        buffer_size=1,
    ):
        self.width = width
        self.height = height
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.autofocus = autofocus
        self.auto_exposure = auto_exposure
        self.fps = fps
        self.buffer_size = buffer_size


class VideoCaptureWrapper:
    def __init__(self, video_path, config: CameraConfig):
        self.cap = cv2.VideoCapture(video_path)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, config.buffer_size)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, config.brightness)
        self.cap.set(cv2.CAP_PROP_CONTRAST, config.contrast)
        self.cap.set(cv2.CAP_PROP_SATURATION, config.saturation)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, config.autofocus)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, config.auto_exposure)
        self.cap.set(cv2.CAP_PROP_FPS, config.fps)

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()


class FrameProcessor:
    def enhance(self, frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], 1.5)  # type: ignore
        hsv[:, :, 2] = cv2.multiply(hsv[:, :, 2], 1.2)  # type: ignore
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


class ShotDetector:
    def __init__(self, capture_frames: int):
        self.capture_frames = capture_frames
        self.capturing = False
        self.captured_frames = []

    def should_start_capture(self, detections):
        return bool(detections)

    def should_stop_capture(self):
        return len(self.captured_frames) >= self.capture_frames

    def reset(self):
        self.capturing = False
        self.captured_frames = []


class VideoListener:
    def on_frame(self, frame):
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False
        return True


class VideoService:
    def __init__(
        self,
        detection_service: BasketballDetectionService,
        video_model: ModelWrapperBase,
        capture_frames: int = 70,
        frame_processor=None,
        video_listener=None,
    ):
        self.detection_service = detection_service
        self.capture_frames = capture_frames
        self.video_model = video_model
        self.frame_processor = frame_processor or FrameProcessor()
        self.video_listener = video_listener or VideoListener()

    def process_video(self, video_path: str | int = 0) -> list[VideoResponse]:
        config = CameraConfig()
        cap = VideoCaptureWrapper(video_path, config)
        shot_detector = ShotDetector(self.capture_frames)
        decisions = []
        shot_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if not shot_detector.capturing:
                detections = self.detection_service.detect_objects(frame)
                if shot_detector.should_start_capture(detections):
                    shot_detector.capturing = True
                    shot_detector.captured_frames = []
            else:
                enhanced = self.frame_processor.enhance(frame.copy())
                shot_detector.captured_frames.append(enhanced)
                if shot_detector.should_stop_capture():
                    shot_count += 1
                    decision = self.video_model.llm_predict(
                        shot_detector.captured_frames
                    )
                    decisions.append(decision)
                    shot_detector.reset()
            if not self.video_listener.on_frame(frame):
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
