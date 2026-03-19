import cv2
import os

import json
import numpy as np
import time

from dotenv import load_dotenv
from google import genai
from typing import List, Tuple
from .base_model import ModelWrapperBase
from .models import VideoResponse
from ..utils import (
    logger,
    VideoWriter, 
    TempFileVideosManager
)
from .prompt_provider import PromptProvider


load_dotenv()


GEMINI_API_KEY = os.getenv("API_KEY_GEMINI")
PROJECT_ID = os.getenv("GEMINI_PROJECT_ID")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "")


class ErrorCaseHandler:
    """Gestiona casos de error predefinidos para respuestas del modelo."""
    
    ERROR_CASES = {
        "no_frames": {
            "reasoning": "No hay frames válidos para procesar",
            "result": "INDETERMINADO",
            "confidence_percentage": 0.0,
        },
        "processing_failed": {
            "reasoning": "El proceso falló",
            "result": "INDETERMINADO",
            "confidence_percentage": 0.0,
        },
        "invalid_json": {
            "reasoning": "Gemini no devolvió un JSON válido.",
            "result": "INDETERMINADO",
            "confidence_percentage": 0.0,
        },
        "exception": {
            "reasoning": "Ocurrió un error en el proceso.",
            "result": "INDETERMINADO",
            "confidence_percentage": 0.0,
        },
    }
    
    @classmethod
    def get_error_case(cls, error_key: str) -> dict:
        """Obtiene un caso de error por su clave."""
        return cls.ERROR_CASES.get(error_key, cls.ERROR_CASES["exception"])
    
    @classmethod
    def add_error_case(cls, error_key: str, error_data: dict) -> None:
        """Añade un nuevo caso de error (extensibilidad)."""
        cls.ERROR_CASES[error_key] = error_data


class GoogleModelClient():

    def __init__(self, ):

        self.client = genai.Client(api_key = GEMINI_API_KEY, project = PROJECT_ID)

    def upload_video(self, video_path: str):

        logger.info("Subiendo video a Gemini File API...")
        video_file = self.client.files.upload(file=video_path)
        return video_file
    
    def check_video_processing(self, video_file: genai.types.File):

        while video_file.state.name == "PROCESSING": # type: ignore
            logger.debug("⏳ Esperando procesamiento del video en los servidores de Google...")
            time.sleep(1)
            video_file = self.client.files.get(name=video_file.name) # type: ignore

        if video_file.state.name == "FAILED": # type: ignore
            logger.error("Falló el procesamiento del video en Gemini.")
            return False
        
        return True
    
    def analyze_video(self, video_file: genai.types.File, prompt: str = PromptProvider.get_prompt()) -> genai.types.GenerateContentResponse:
        logger.info("Analizando video...")
        response = self.client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=[video_file, prompt]
        )

        self.client.files.delete(name=video_file.name) # type: ignore

        return response
    
    def parse_response(self, response: genai.types.GenerateContentResponse) -> VideoResponse:
        try:
            if not response or not response.text:
                logger.warning("Gemini no devolvió una respuesta válida.")
                return VideoResponse(**ErrorCaseHandler.get_error_case("processing_failed"))

            result_text = response.text.strip().replace("```json", "").replace("```", "")
            response_json = json.loads(result_text)

            logger.debug(f"Razonamiento: {response_json.get('razonamiento')}")
            return VideoResponse(**response_json)
        except json.JSONDecodeError:
            logger.error("Gemini no devolvió un JSON válido.")
            return VideoResponse(**ErrorCaseHandler.get_error_case("invalid_json"))
        except Exception as e:
            logger.error(f"Ocurrió un error al procesar la respuesta de Gemini: {e}")
            return VideoResponse(**ErrorCaseHandler.get_error_case("exception"))


class GoogleModelConfig:
    """Agrupa las dependencias de GoogleModelWrapper en un único objeto."""
    
    def __init__(self, 
                 google_client: GoogleModelClient,
                 video_writer: VideoWriter,
                 temp_file_manager: TempFileVideosManager):
        self.google_client = google_client
        self.video_writer = video_writer
        self.temp_file_manager = temp_file_manager


class GoogleModelWrapper(ModelWrapperBase):

    _FPS = 4.0

    def __init__(self, config: GoogleModelConfig):

        self.client = config.google_client
        self.video_writer = config.video_writer
        self.temp_file_manager = config.temp_file_manager

    def __frames_evaluation_shape(self, frames: List[np.ndarray]) -> Tuple[int, int]:

        if len(frames) > 60:
            sampled_indices = list(range(0, len(frames), 3))
            sampled_frames = [frames[i] for i in sampled_indices]
        else:
            sampled_frames = frames
    
        return sampled_frames[0].shape

    def __predict(self, frames: List[np.ndarray]) -> VideoResponse:

        if not frames or not isinstance(frames, list) or len(frames) == 0:
            logger.warning("No hay frames válidos para procesar")
            return VideoResponse(**ErrorCaseHandler.get_error_case("no_frames"))

        logger.info(f"Procesando secuencia de {len(frames)} frames en video MP4...")
        sampled_frames_shape = self.__frames_evaluation_shape(frames)

        self.temp_file_manager.set_names()
        self.video_writer.file_path = self.temp_file_manager.original_video_path
        self.video_writer.write_video(frames, fps=self._FPS, height=sampled_frames_shape[0], width=sampled_frames_shape[1])

        logger.info("Subiendo video a Gemini File API...")
        video_file = self.client.upload_video(self.video_writer.file_path)

        video_ready = self.client.check_video_processing(video_file)
        if not video_ready:
            return VideoResponse(**ErrorCaseHandler.get_error_case("processing_failed"))
        
        response = self.client.analyze_video(video_file)
        parsed_response = self.client.parse_response(response)
        parsed_response.video_name = self.temp_file_manager.original_file  
        return parsed_response

    def llm_predict(self, frame, *args, **kwargs) -> VideoResponse:
        return self.__predict(frame, *args, **kwargs)


class GoogleModelWrapperBuilder:
    """Builder para construir GoogleModelWrapper de forma fluida."""
    
    def __init__(self):
        self._google_client = None
        self._video_writer = None
        self._temp_manager = None
    
    def set_google_client(self, client: GoogleModelClient):
        self._google_client = client
        return self
    
    def set_video_writer(self, writer: VideoWriter):
        self._video_writer = writer
        return self
    
    def set_temp_manager(self, manager: TempFileVideosManager):
        self._temp_manager = manager
        return self
    
    def build(self) -> GoogleModelWrapper:
        config = GoogleModelConfig(
            google_client=self._google_client, # type: ignore
            video_writer=self._video_writer, # type: ignore
            temp_file_manager=self._temp_manager # type: ignore
        )
        return GoogleModelWrapper(config)


if __name__ == "__main__":
    pass
