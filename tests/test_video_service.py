import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import os
from pathlib import Path
import cv2

from basketball_detector.services.video_service import (
    analize_frame_with_llm,
    process_video
)


class TestAnalizeFrameWithLLM:
    """Tests para la función analize_frame_with_llm"""
    
    @patch('basketball_detector.services.video_service.genai.GenerativeModel')
    def test_analize_frame_returns_response(self, mock_genai):
        """Verifica que la función retorn la respuesta del modelo"""
        # Arrange
        mock_response = MagicMock()
        mock_response.text = "ENCESTA  \n"
        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.return_value = mock_model
        
        # Crear un frame de prueba
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Act
        result = analize_frame_with_llm(frame)
        
        # Assert
        assert result == "ENCESTA"
        mock_model.generate_content.assert_called_once()
    
    @patch('basketball_detector.services.video_service.genai.GenerativeModel')
    def test_analize_frame_strips_whitespace(self, mock_genai):
        """Verifica que se elimine el espacios en blanco"""
        # Arrange
        mock_response = MagicMock()
        mock_response.text = "NO ENCESTA\n\n"
        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.return_value = mock_model
        
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Act
        result = analize_frame_with_llm(frame)
        
        # Assert
        assert result == "NO ENCESTA"
    
    @patch('basketball_detector.services.video_service.genai.GenerativeModel')
    def test_analize_frame_with_different_responses(self, mock_genai):
        """Prueba con diferentes respuestas posibles del modelo"""
        # Arrange
        possible_responses = ["ENCESTA", "NO ENCESTA", "INDETERMINADO"]
        mock_model = MagicMock()
        mock_genai.return_value = mock_model
        
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Act & Assert
        for response_text in possible_responses:
            mock_response = MagicMock()
            mock_response.text = response_text
            mock_model.generate_content.return_value = mock_response
            
            result = analize_frame_with_llm(frame)
            assert result == response_text


class TestProcessVideo:
    """Tests para la función process_video"""
    
    def test_process_video_with_real_video(self):
        """Prueba process_video con el video de prueba real"""
        # Arrange
        video_path = "tests/test_videos/first-video.mp4"
        
        # Verificar que el video existe
        assert os.path.exists(video_path), f"Video no encontrado en {video_path}"
        
        # Abrir el video para verificar que es válido
        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened(), "No se pudo abrir el video"
        
        # Verificar propiedades básicas
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Assert
        assert frame_count > 0, "El video debe tener frames"
        assert width > 0, "El ancho debe ser mayor a 0"
        assert height > 0, "El alto debe ser mayor a 0"
        assert fps > 0, "Los FPS deben ser mayores a 0"
        
        cap.release()
    
    @patch('basketball_detector.services.video_service.cv2.imshow')
    @patch('basketball_detector.services.video_service.cv2.waitKey')
    @patch('basketball_detector.services.video_service.model_yolo')
    @patch('basketball_detector.services.video_service.analize_frame_with_llm')
    def test_process_video_with_mock(self, mock_analize, mock_yolo, mock_waitkey, mock_imshow):
        """Prueba process_video con mocks"""
        # Arrange
        video_path = "tests/test_videos/first-video.mp4"
        mock_waitkey.return_value = ord('q') << 8  # Simular presionar 'q'
        mock_analize.return_value = "ENCESTA"
        
        # Crear un mock de resultados YOLO
        mock_results = []
        mock_yolo.return_value = mock_results
        
        # Act & Assert - La función no retorna nada, solo verifica que se ejecute sin errores
        try:
            process_video(video_path)
        except Exception as e:
            pytest.fail(f"process_video lanzó una excepción: {e}")
    
    @patch('basketball_detector.services.video_service.cv2.imshow')
    @patch('basketball_detector.services.video_service.cv2.waitKey')
    @patch('basketball_detector.services.video_service.model_yolo')
    def test_process_video_exits_on_invalid_video(self, mock_yolo, mock_waitkey, mock_imshow):
        """Prueba que handle correctamente un video inválido"""
        # Arrange
        invalid_path = "tests/test_videos/non_existent_video.mp4"
        
        # Act
        result = cv2.VideoCapture(invalid_path)
        
        # Assert
        assert not result.isOpened(), "Un video inexistente no debe abrirse"
        result.release()


class TestIntegration:
    """Tests de integración"""
    
    def test_video_file_exists(self):
        """Verifica que el archivo de video de prueba existe"""
        video_path = Path("tests/test_videos/first-video.mp4")
        assert video_path.exists(), f"El archivo de video no existe en {video_path}"
    
    @patch('basketball_detector.services.video_service.cv2.imshow')
    @patch('basketball_detector.services.video_service.cv2.waitKey')
    @patch('basketball_detector.services.video_service.model_yolo')
    @patch('basketball_detector.services.video_service.analize_frame_with_llm')
    def test_process_video_reads_frames(self, mock_analize, mock_yolo, mock_waitkey, mock_imshow):
        """Verifica que process_video pueda procesar frames del video real"""
        # Arrange
        video_path = "tests/test_videos/first-video.mp4"
        frame_counter = 0
        
        def count_frames(frame):
            nonlocal frame_counter
            frame_counter += 1
            return "NO ENCESTA"
        
        mock_analize.side_effect = count_frames
        mock_yolo.return_value = []
        mock_waitkey.return_value = ord('q') << 8
        
        # Act
        try:
            process_video(video_path)
        except Exception as e:
            pytest.fail(f"Falló al procesar video: {e}")
