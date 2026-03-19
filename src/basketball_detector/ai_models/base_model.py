from abc import ABC, abstractmethod

from ..models.detection_model import VideoResponse


class ModelWrapperBase(ABC):
    """Clase base para todos los wrappers de modelos de IA."""
    
    @abstractmethod
    def llm_predict(self, frames, *args, **kwargs) -> VideoResponse:
        """
        Predice si una canasta fue encestada.
        
        Args:
            frames: Array de imágenes (frames) o ruta a video
            
        Returns:
            VideoResponse: Respuesta con resultado y confianza
        """
        pass
