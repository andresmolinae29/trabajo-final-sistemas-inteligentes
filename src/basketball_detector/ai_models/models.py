import json
from abc import ABC, abstractmethod

from basketball_detector.utils.manage_temp_files import TempFileVideosManager
from basketball_detector.utils.video_writer import VideoWriter

from ..models.detection_model import VideoResponse
from .base_model import ModelWrapperBase
from .google_model import GoogleModelClient, GoogleModelWrapperBuilder, GoogleModelWrapper


class GoogleVideoModel(ModelWrapperBase):
    def __init__(self, wrapper: GoogleModelWrapper):
        self.wrapper = wrapper

    def llm_predict(self, frames, *args, **kwargs) -> VideoResponse:
        return self.wrapper.llm_predict(frames, *args, **kwargs)


class GoogleVideoModelFactory:
    """Crea GoogleVideoModel con todas sus dependencias."""
    
    @staticmethod
    def create() -> ModelWrapperBase:
        client = GoogleModelClient()
        manager = TempFileVideosManager()
        writer = VideoWriter(manager.temp_video_path)
        
        wrapper = (GoogleModelWrapperBuilder()
            .set_google_client(client)
            .set_video_writer(writer)
            .set_temp_manager(manager)
            .build())
        
        return GoogleVideoModel(wrapper)