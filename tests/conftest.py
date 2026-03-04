"""Configuración de pytest para mockear dependencias problemáticas"""
import sys
from unittest.mock import MagicMock

# Mock de dependencias problemáticas ANTES de que se importen
sys.modules['ultralytics'] = MagicMock()
sys.modules['google.generativeai'] = MagicMock()
sys.modules['torch'] = MagicMock()
sys.modules['torchvision'] = MagicMock()
