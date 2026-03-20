# 🏀 Basketball Shot Detector

Sistema inteligente de detección de canastas de baloncesto que utiliza visión por computadora e inteligencia artificial para analizar videos y determinar si una canasta fue encestada exitosamente.

## 📋 Descripción General

Este proyecto combina **detección de objetos en tiempo real** (YOLO) con **análisis de video por inteligencia artificial** (Gemini) para crear un sistema completo de análisis automático de jugadas de baloncesto. La aplicación procesa videos, identifica momentos clave y proporciona un análisis detallado con confianza en las predicciones.

## 🎯 Característica Principales

- ✅ Detección automática de balones de baloncesto en videos
- ✅ Análisis inteligente de canastas usando IA generativa (Gemini)
- ✅ Interfaz web con FastAPI para cargar y procesar videos
- ✅ Segmentación de videos en momentos clave
- ✅ Confianza y razonamiento explícito de cada decisión
- ✅ Gestión automática de archivos temporales
- ✅ Logging detallado de procesos

## 🏗️ Arquitectura del Proyecto

```
basketball_detector/
├── ai_models/          # Modelos de inteligencia artificial
│   ├── base_model.py   # Clase abstracta base
│   ├── google_model.py # Wrapper para Gemini API
│   └── models.py       # Factories y coordinadores
├── models/             # Modelos de datos (Pydantic)
│   └── detection_model.py
├── services/           # Lógica de negocio
│   ├── detection_service.py
│   └── video_service.py
├── utils/              # Utilidades
│   ├── logger.py
│   ├── manage_temp_files.py
│   └── video_writer.py
├── static/             # Archivos estáticos (CSS, videos)
└── templates/          # Templates HTML
```

## 🔧 Clases Principales

### **ai_models/** - Modelos de IA

#### `ModelWrapperBase` (base_model.py)
Clase abstracta que define la interfaz para todos los wrappers de modelos.

```python
class ModelWrapperBase(ABC):
    @abstractmethod
    def llm_predict(self, frames, *args, **kwargs) -> VideoResponse:
        """Predice si una canasta fue encestada."""
        pass
```

**Responsabilidades:**
- Define contrato para predicciones de modelos de IA
- Retorna `VideoResponse` con resultado, confianza y razonamiento

---

#### `GoogleModelClient` (google_model.py)
Maneja la comunicación con la API de Google Gemini.

**Métodos clave:**
- `upload_video(video_path)` - Sube video a Gemini Files API
- `check_video_processing(video_file)` - Espera a que Gemini procese el video
- `analyze_video(video_file, prompt)` - Envía prompt y obtiene análisis

**Características:**
- Gestiona claves API desde variables de entorno
- Monitorea estado de procesamiento en servidores de Google
- Limpia archivos después del análisis

---

#### `GoogleModelWrapper`
Wrapper que orquesta el flujo completo de análisis.

**Responsabilidades:**
- Coordina subida, procesamiento y análisis de videos
- Parsea respuestas JSON de Gemini
- Maneja errores y casos especiales
- Genera mini-videos con anotaciones de decisiones

---

#### `GoogleVideoModel` & `GoogleVideoModelFactory`
Implementación concreta del patrón Factory.

**Factory Pattern:**
```python
GoogleVideoModelFactory.create() -> ModelWrapperBase
```

Centraliza la creación de `GoogleVideoModel` con todas sus dependencias inyectadas.

---

### **services/** - Servicios de Detección

#### `BasketballDetectionService` (detection_service.py)
Orquesta todo el pipeline de detección de objetos.

**Componentes:**

1. **ObjectDetector**
   - Usa modelo YOLOE-26s para detectar balones
   - Retorna lista de `DetectionResult` con bounding boxes
   - Configurable con threshold de confianza (default: 0.05)

2. **ObjectFilter**
   - Filtra detecciones por ubicación espacial
   - `is_in_top_quarter()` - Verifica si el balón está en zona de tiro

3. **ObjectValidator**
   - Valida presencia de objetos específicos
   - `is_valid_detection()` - Confirma si el balón fue detectado

4. **DistanceCalculator**
   - Calcula distancia entre objetos detectados
   - Soporta cálculo en píxeles y centímetros
   - `PIXELS_PER_CM = 4.0` constante de calibración

**Métodos principales:**
```python
detect_objects(frame: np.ndarray) -> list[DetectionResult]
validate_detection(detections: list) -> bool
```

---

#### `VideoService` (video_service.py)
Procesa videos frame a frame y coordina el pipeline completo.

**Componentes:**

1. **CameraConfig**
   - Configura parámetros de captura de video
   - Resolución: 1280x720 (default)
   - FPS: 30 (default)
   - Ajustes: brillo, contraste, saturación

2. **VideoCaptureWrapper**
   - Encapsula `cv2.VideoCapture`
   - Aplica configuración de cámara

3. **FrameProcessor**
   - Mejora calidad de frames antes de procesamiento
   - `enhance()` - Aumenta saturación y brillo (HSV)

4. **ShotDetector**
   - Captura frames cuando detecta movimiento (posible tiro)
   - `capture_frames` - Número de frames a capturar (default: 70)
   - `should_start_capture()` - Inicia captura si hay detecciones
   - `should_stop_capture()` - Para cuando llega al límite

5. **VideoListener**
   - Interface para visualización en tiempo real
   - Permite interrumpir con tecla 'q'

**Método principal:**
```python
process_video(video_path: str) -> list[VideoResponse]
```

---

### **models/** - Modelos de Datos

#### `DetectionResult` (Pydantic)
Resultado de una detección individual.

```python
class DetectionResult(BaseModel):
    label: str                                      # "Basketball ball"
    confidence: float | Any                         # 0.0 - 1.0
    bbox: tuple[float, float, float, float]        # (x1, y1, x2, y2)
```

---

#### `VideoResponse` (Pydantic)
Respuesta final del análisis de un clip de video.

```python
class VideoResponse(BaseModel):
    reasoning: str                     # Explicación detallada
    result: str                        # "CANASTA" | "NO_CANASTA" | "INDETERMINADO"
    confidence_percentage: float       # Porcentaje de confianza
    video_name: str | None             # Nombre del video generado
```

## 🤖 Modelos de IA Utilizados

### **1. YOLOv8 (Ultralytics) - Detección de Objetos**

**Modelo:** `yoloe-26s-seg.pt`

**Propósito:** Detecta balones de baloncesto en frames de video

**Características:**
- Modelo YOLO optimizado para segmentación
- Entrenado específicamente para detectar balones
- Configuración: threshold mínimo de 0.05 para máxima sensibilidad
- Rápido (~30-50ms por frame)
- Soporta GPU con CUDA

**Uso:**
```python
from ultralytics import YOLO
model = YOLO("yoloe-26s-seg.pt")
results = model.predict(frame, conf=0.05)
```

---

### **2. Google Gemini (Google GenAI) - Análisis de Video**

**Modelo:** Configurado en variable de entorno `GEMINI_MODEL_NAME`
- Actualmente: `gemini-3.1-pro-preview`

**Propósito:** Analiza clips de video y determina si fue canasta

**Características:**
- Procesa video con comprensión visual avanzada
- Recibe prompt específico con contexto del análisis
- Retorna análisis en formato JSON estructurado
- Proporciona razonamiento detallado de la decisión

**Flujo:**
1. Sube video a Gemini Files API
2. Espera procesamiento en servidores de Google
3. Envía prompt con instrucciones específicas
4. Parsea respuesta JSON
5. Limpia archivos en servidor

**Variables de Entorno Requeridas:**
```
API_KEY_GEMINI=<tu-clave-api>
PROJECT_ID=<tu-project-id>
GEMINI_MODEL_NAME=gemini-3.1-pro-preview
```

---

## 🔐 Configuración de Variables de Entorno

Crea un archivo `.env` en la raíz del proyecto con las siguientes variables:

```bash
# Google Gemini Configuration
API_KEY_GEMINI="AIzaSyBInDtfG-v_DmNQqzuLCcxj72Oivd32sMY"
PROJECT_ID="gen-lang-client-0404562963"
GEMINI_MODEL_NAME="gemini-3.1-pro-preview"
```

**Descripción:**

| Variable | Descripción |
|----------|-------------|
| `API_KEY_GEMINI` | Clave de API de Google Cloud para autenticación |
| `PROJECT_ID` | ID del proyecto de Google Cloud |
| `GEMINI_MODEL_NAME` | Nombre del modelo Gemini a usar (versión de IA generativa) |

**Cómo obtenerlas:**
1. Ve a [Google Cloud Console](https://console.cloud.google.com/)
2. Crea un proyecto o selecciona uno existente
3. Activa la API de Gemini
4. Crea credenciales (API Key)
5. Copia los valores al archivo `.env`

---

## 📦 Dependencias Principales

```toml
# Detección de Objetos
ultralytics>=8.4.17    # YOLOv8 para detección
torch>=2.10.0          # Framework de deep learning
torchvision>=0.25.0    # Utilidades de visión para PyTorch

# Procesamiento de Video
opencv-python>=4.13.0  # Captura y procesamiento de frames
pillow>=12.1.1         # Manipulación de imágenes

# APIs e IA
google-genai>=1.65.0   # Google Gemini API
transformers>=5.3.0    # Modelos pre-entrenados

# Web Framework
fastapi>=0.133.1       # Framework web moderno
uvicorn>=0.41.0        # Servidor ASGI

# Optimización
bitsandbytes>=0.49.2   # Cuantización de modelos
accelerate>=1.13.0     # Aceleración de transformers

# Utilidades
python-multipart>=0.0.22  # Manejo de uploads
dotenv>=0.9.9             # Gestión de variables de entorno
pytest>=9.0.2             # Testing
```

---

## 🚀 Instalación

### Requisitos Previos
- Python 3.11 - 3.14
- Poetry (gestor de dependencias)
- GPU con CUDA (recomendado para YOLOv8)

### Pasos

1. **Clonar el repositorio**
```bash
git clone <url-del-repo>
cd trabajo-final-sistemas-inteligentes
```

2. **Instalar dependencias con Poetry**
```bash
poetry install
```

3. **Configurar variables de entorno**
```bash
# Crear archivo .env
cp .env.example .env
# Editar .env con tus credenciales de Gemini
```

4. **Descargar modelos pre-entrenados** (opcional, se descargan automáticamente)
```bash
# YOLOv8 se descargará en primer uso
# Modelos disponibles:
# - yoloe-26s-seg.pt (actual)
# - yolov8n.pt (alternativa ligera)
```

---

## ▶️ Ejecución

### Iniciar la aplicación web

```bash
poetry run python src/main.py
```

O de forma equivalente:
```bash
cd src
poetry run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Parámetros:**
- `--host 0.0.0.0` - Accesible desde cualquier IP
- `--port 8000` - Puerto de escucha
- `--reload` - Reinicia en cambios de código (desarrollo)

La aplicación estará disponible en: **http://localhost:8000**

---

## 🔄 Flujo de Procesamiento

```
Video de entrada
    ↓
[VideoCaptureWrapper] - Captura frames
    ↓
[FrameProcessor] - Mejora calidad
    ↓
[ObjectDetector (YOLOv8)] - Detecta balones
    ↓
[ShotDetector] - ¿Hay tiro? Captura 70 frames
    ↓
[VideoWriter] - Genera mini-video
    ↓
[GoogleModelClient] - Sube a Gemini
    ↓
[GoogleModelWrapper] - Envía prompt
    ↓
[Gemini LLM] - Analiza y retorna JSON
    ↓
[VideoResponse] - Resultado final
    ↓
Página web con resultados
```

---

## 📊 Resultados de Análisis

Cada análisis retorna un objeto `VideoResponse` con:

```json
{
  "resultado": "CANASTA",
  "confianza_porcentaje": 95.2,
  "razonamiento": "Se detectó movimiento del balón en trayectoria hacia la canasta...",
  "nombre_video": "mini_video_001.mp4"
}
```

**Posibles resultados:**
- `CANASTA` - La canasta fue encestada exitosamente
- `NO_CANASTA` - El tiro no entró o no se detectó
- `INDETERMINADO` - No hay suficiente información

---

## 📝 Estructura de Logs

La aplicación genera logs detallados en consola:

```
[INFO] Subiendo video a Gemini File API...
[DEBUG] ⏳ Esperando procesamiento del video en los servidores de Google...
[INFO] Analizando video...
[INFO] Decisiones obtenidas: [VideoResponse(...), ...]
```

Ver [logger.py](src/basketball_detector/utils/logger.py) para configuración.

---

## 🛠️ Desarrollo

### Estructura de Carpetas de Utilidades

- **logger.py** - Sistema de logging centralizado
- **manage_temp_files.py** - Gestión de archivos temporales
- **video_writer.py** - Escritura y procesamiento de videos

### Patrones de Diseño Utilizados

1. **Factory Pattern** - `GoogleVideoModelFactory`, `DetectionServiceFactory`
2. **Strategy Pattern** - `ModelWrapperBase` con diferentes implementaciones
3. **Builder Pattern** - `GoogleModelWrapperBuilder`
4. **Dependency Injection** - En constructores de servicios

---

## 🤝 Contribuciones

Para contribuir:

1. Fork el repositorio
2. Crea una rama (`git checkout -b feature/AmazingFeature`)
3. Commit cambios (`git commit -m 'Add AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver archivo `LICENSE` para detalles.

---

## 👤 Autor

**Andrés Molina**
- Email: andres.molinae29@gmail.com
- Proyecto Final de Sistemas Inteligentes

---

## 📞 Soporte

Para reportar bugs o solicitar features, abre un issue en el repositorio.

---

**Última actualización:** Marzo 2026
