from ultralytics import YOLO, YOLOE
from basketball_detector.models.detection_model import DetectionResult


MODEL: YOLOE = YOLO("yoloe-26s-seg.pt") # type: ignore
PIXELS_PER_CM = 4.0
CONFIDENCE_THRESHOLDS = {
    "Basketball ball": 0.05,
    # "Basketball hoop": 0.05
}


class DetectionService:
    OBJECTS = list(CONFIDENCE_THRESHOLDS.keys())
    PROXIMITY_THRESHOLD_CM = 20.0

    def __init__(self, model: YOLOE = MODEL):
        self.model = model
        self.model.set_classes(self.OBJECTS)

    def _is_in_top_quarter(self, bbox: tuple[int, int, int, int], image_height: int) -> bool:
        _, y1, _, y2 = bbox
        center_y = (y1 + y2) / 2
        return center_y <= image_height / 4

    def detect_objects(self, frame, conf_threshold: float = 0.05, top_quarter_only: bool = True) -> list[DetectionResult]:
        """
        Detecta objetos en el frame aplicando umbrales de confianza específicos por objeto.
        
        Args:
            frame: Frame de video
            image_height: Altura de la imagen
            conf_threshold: Umbral global mínimo (se usa config por clase si es mayor)
            
        Returns:
            Lista de detecciones filtradas por confianza
        """
        results = self.model.predict(frame, conf=conf_threshold)
        detections = []

        image_height = None
        if hasattr(frame, "shape") and len(frame.shape) >= 2:
            image_height = frame.shape[0]
        elif hasattr(frame, "height"):
            image_height = frame.height
        
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                class_id = int(box.cls[0])
                label = self.OBJECTS[class_id]

                if top_quarter_only and image_height is not None:
                    if not self._is_in_top_quarter((x1, y1, x2, y2), image_height):
                        continue

                if label in [d.label for d in detections]:
                    continue

                detections.append(
                    DetectionResult(
                        label=label,
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2),
                    )
                )
        
        return detections
    
    def balled_ball_in_frame(self, detections: list[DetectionResult]) -> bool:
        """Verifica si el balón está presente en la imagen."""
        return any(d.label == "Basketball ball" for d in detections)
    
    def hoop_and_ball_in_frame(self, detections: list[DetectionResult]) -> bool:
        """Verifica si tanto el balón como el aro están presentes en la imagen."""
        labels = [d.label for d in detections]
        return all(obj in labels for obj in self.OBJECTS)

    def distance_between_two_objects(
        self, obj1: DetectionResult, obj2: DetectionResult
    ) -> float:
        """Calcula la distancia en PÍXELES entre dos objetos."""
        x1_1, y1_1, x2_1, y2_1 = obj1.bbox
        x1_2, y1_2, x2_2, y2_2 = obj2.bbox
        center_x1 = (x1_1 + x2_1) / 2
        center_y1 = (y1_1 + y2_1) / 2
        center_x2 = (x1_2 + x2_2) / 2
        center_y2 = (y1_2 + y2_2) / 2
        distance = ((center_x1 - center_x2) ** 2 + (center_y1 - center_y2) ** 2) ** 0.5
        return distance
    
    def distance_between_two_objects_cm(
        self, obj1: DetectionResult, obj2: DetectionResult
    ) -> float:
        """Calcula la distancia en CENTÍMETROS entre dos objetos."""
        distance_pixels = self.distance_between_two_objects(obj1, obj2)
        return distance_pixels / PIXELS_PER_CM
    
    def are_two_objects_close(
        self, obj1: DetectionResult, obj2: DetectionResult,
        threshold_cm: float = PROXIMITY_THRESHOLD_CM
    ) -> bool:
        """
        Verifica si dos objetos están cerca (para iniciar evaluación).
        Retorna True si están a menos de 'threshold_cm' centímetros.
        """
        distance_cm = self.distance_between_two_objects_cm(obj1, obj2)
        return distance_cm < threshold_cm


if __name__ == "__main__":
    pass
