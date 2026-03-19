import numpy as np

from ultralytics import YOLO, YOLOE
from basketball_detector.models.detection_model import DetectionResult


MODEL: YOLOE = YOLO("yoloe-26s-seg.pt") # type: ignore
PIXELS_PER_CM = 4.0


class ObjectDetector:

    def __init__(self, model: YOLOE) -> None:
        self.model = model

    def detect_objects(self, frame: np.ndarray, conf_threshold: float = 0.05) -> list[DetectionResult]:
        results = self.model.predict(frame, conf=conf_threshold)
        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                class_id = int(box.cls[0])
                label = self.model.names[class_id]

                detections.append(
                    DetectionResult(
                        label=label,
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2),
                    )
                )
        
        return detections
    

class ObjectFilter:

    @staticmethod
    def is_in_top_quarter(bbox: tuple[float, float, float, float], image_height: int) -> bool:
        _, y1, _, y2 = bbox
        center_y = (y1 + y2) / 2
        return center_y <= image_height / 4
    

class ObjectValidator:

    @staticmethod
    def is_valid_detection(detections: list[DetectionResult], target_label: str) -> bool:
        """Verifica si el objeto objetivo está presente en la lista de detecciones."""
        return any(d.label == target_label for d in detections)
    

class DistanceCalculator:

    @staticmethod
    def distance_between_two_objects(obj1: DetectionResult, obj2: DetectionResult) -> float:
        """Calcula la distancia en PÍXELES entre dos objetos."""
        x1_1, y1_1, x2_1, y2_1 = obj1.bbox
        x1_2, y1_2, x2_2, y2_2 = obj2.bbox
        center_x1 = (x1_1 + x2_1) / 2
        center_y1 = (y1_1 + y2_1) / 2
        center_x2 = (x1_2 + x2_2) / 2
        center_y2 = (y1_2 + y2_2) / 2
        distance = ((center_x1 - center_x2) ** 2 + (center_y1 - center_y2) ** 2) ** 0.5
        return distance
    
    @staticmethod
    def distance_between_two_objects_cm(obj1: DetectionResult, obj2: DetectionResult) -> float:
        """Calcula la distancia en CENTÍMETROS entre dos objetos."""
        distance_pixels = DistanceCalculator.distance_between_two_objects(obj1, obj2)
        return distance_pixels / PIXELS_PER_CM


class DetectionServiceConfig:
    """Configura las dependencias del DetectionService."""
    
    def __init__(self, 
                 object_detector: ObjectDetector, 
                 object_filter: ObjectFilter, 
                 object_validator: ObjectValidator, 
                 distance_calculator: DistanceCalculator):
        self.object_detector = object_detector
        self.object_filter = object_filter
        self.object_validator = object_validator
        self.distance_calculator = distance_calculator


class BasketballDetectionService:

    def __init__(self, config: DetectionServiceConfig):
        self.object_detector = config.object_detector
        self.object_filter = config.object_filter
        self.object_validator = config.object_validator
        self.distance_calculator = config.distance_calculator

    def detect_objects(self, frame: np.ndarray) -> list[DetectionResult]:
        objects = self.object_detector.detect_objects(frame)

        if self.object_validator.is_valid_detection(objects, target_label="Basketball ball") \
            and self.object_filter.is_in_top_quarter(objects[0].bbox, frame.shape[0]):
            return objects
        return []
    

class DetectionServiceFactory:
    """Fábrica para crear instancias de BasketballDetectionService con todas sus dependencias."""
    
    @staticmethod
    def create() -> BasketballDetectionService:
        object_detector = ObjectDetector(MODEL)
        object_filter = ObjectFilter()
        object_validator = ObjectValidator()
        distance_calculator = DistanceCalculator()

        config = DetectionServiceConfig(
            object_detector=object_detector,
            object_filter=object_filter,
            object_validator=object_validator,
            distance_calculator=distance_calculator
        )

        return BasketballDetectionService(config)

    
if __name__ == "__main__":
    pass
