# Información para Presentación PPT - Basketball Shot Detector

---

## 📊 SLIDE 1: MOTIVACIÓN DEL TRABAJO

### Título
**"Análisis Automatizado de Canastas de Baloncesto con IA"**

### Contenido - Puntos Clave

1. **Problema Actual**
   - Análisis manual de videos de baloncesto consume mucho tiempo
   - Decisiones arbitradas en tiempo real pueden ser subjetivas
   - Necesidad de automatización en deportes de élite

2. **Solución Propuesta**
   - Sistema inteligente que detecta y analiza canastas automáticamente
   - Combina visión por computadora (YOLO) con IA generativa (Gemini)
   - Proporciona análisis explícito con confianza en las decisiones

3. **Beneficios**
   - ⚡ **Velocidad**: Procesamiento automático de videos
   - 🎯 **Precisión**: Detección de objetos + análisis de IA
   - 📊 **Explicabilidad**: Razonamiento detallado de cada decisión
   - 🔄 **Escalabilidad**: Puede procesar múltiples videos en paralelo

4. **Aplicaciones**
   - Análisis de partidos profesionales
   - Entrenamiento de equipos
   - Estadísticas automáticas de jugadores
   - Generación de highlights automáticos

---

## 🏗️ SLIDE 2: ARQUITECTURA DEL PROYECTO Y MODELOS USADOS

### Título
**"Arquitectura General y Modelos de IA"**

### Diagrama de Flujo (Describe así en la slide)

```
┌─────────────┐
│  Video MP4  │
└──────┬──────┘
       │
       ▼
┌─────────────────────────┐
│ Captura de Frames (OpenCV) │
└──────┬──────────────────┘
       │
       ▼
┌──────────────────────────┐
│ Mejora de Calidad (HSV)  │
└──────┬───────────────────┘
       │
       ▼
┌────────────────────────────────────┐
│ Detección YOLOv8 (Balones)        │
└──────┬─────────────────────────────┘
       │
       ▼
┌────────────────────────────────────┐
│ ShotDetector (Captura 70 frames)  │
└──────┬─────────────────────────────┘
       │
       ▼
┌─────────────────────────┐
│ VideoWriter (Mini-video)│
└──────┬──────────────────┘
       │
       ▼
┌──────────────────────────┐
│ Gemini API (Análisis IA) │
└──────┬───────────────────┘
       │
       ▼
┌──────────────────────────┐
│ VideoResponse (Resultado)│
└──────────────────────────┘
```

### Modelos Utilizados (Tabla)

| Modelo | Propósito | Framework | Ubicación |
|--------|-----------|-----------|-----------|
| **YOLOv8 (yoloe-26s-seg.pt)** | Detectar balones | Ultralytics/PyTorch | GPU/CPU |
| **Gemini 3.1 Pro** | Analizar si fue canasta | Google GenAI | Cloud |

### Stack Tecnológico

**Frontend:**
- HTML/CSS (Bootstrap)
- FastAPI Templates

**Backend:**
- FastAPI (framework web)
- Uvicorn (servidor ASGI)

**IA y Visión:**
- YOLO v8 (detección)
- Google Gemini (LLM + visión)
- OpenCV (procesamiento de video)
- PyTorch (transformers)

**Infraestructura:**
- Python 3.11-3.14
- Poetry (gestor de dependencias)

---

## 🤖 SLIDE 3: PROFUNDIZACIÓN EN MODELOS USADOS

### Título
**"Modelos de IA: YOLOv8 y Google Gemini"**

---

### PARTE A: YOLOv8 - Detección de Balones

#### ¿Qué es?
- **YOLO** = "You Only Look Once"
- Red neuronal convolucional para detección de objetos
- Versión utilizada: **YOLOv8 con segmentación (yoloe-26s-seg.pt)**

#### Características Técnicas
- **Velocidad**: 30-50ms por frame (~30 FPS)
- **Precisión**: Entrenado específicamente para detectar balones
- **Hardware**: Compatible con GPU (CUDA) y CPU
- **Threshold**: 0.05 (sensibilidad máxima para no perder detecciones)

#### Proceso en el Proyecto
1. Recibe frame de video (1280x720)
2. Detecta objetos de clase "Basketball ball"
3. Retorna bounding boxes + confianza
4. Filtra por ubicación (zona de tiro)

#### Output
```json
{
  "label": "Basketball ball",
  "confidence": 0.95,
  "bbox": [x1, y1, x2, y2]
}
```

#### Ventajas
✅ Procesamiento en tiempo real
✅ Múltiples detecciones por frame
✅ Eficiente energéticamente
✅ Altamente preciso para objetos específicos

---

### PARTE B: Google Gemini 3.1 Pro - Análisis de Canastas

#### ¿Qué es?
- **Modelo multimodal** de IA generativa de Google
- Combina comprensión de imágenes + procesamiento de lenguaje natural
- Última generación de IA con razonamiento avanzado

#### Características Técnicas
- **Modalidades**: Video + Texto
- **Entrada**: Video MP4 + Prompt en lenguaje natural
- **Salida**: JSON estructurado con análisis
- **Razonamiento**: Explica por qué fue o no canasta

#### Proceso en el Proyecto
1. Sube video a Google Gemini Files API
2. Espera procesamiento en servidores de Google (~2-3 minutos)
3. Envía prompt con preguntas específicas
4. Gemini analiza framewise el video
5. Retorna JSON con decisión + razonamiento

#### Prompt Típico
```
"Analiza este video de baloncesto. ¿Se metió la canasta?
- Observa la trayectoria del balón
- Mira si entra por la canasta
- Proporciona tu análisis detallado"
```

#### Output
```json
{
  "resultado": "CANASTA",
  "confianza_porcentaje": 92.5,
  "razonamiento": "Se observa claramente la trayectoria del balón 
                   entrando por la canasta en el frame 23..."
}
```

#### Ventajas
✅ Comprensión contextual superior
✅ Explica razonamiento en lenguaje natural
✅ Maneja ambigüedades mejor que métodos clásicos
✅ Aprende de patrones complejos sin reentrenamiento

---

### COMPARACIÓN: YOLOv8 vs Gemini

| Aspecto | YOLOv8 | Gemini |
|--------|--------|--------|
| **Propósito** | Detección de objetos | Análisis semántico |
| **Entrada** | Frame individual | Secuencia de frames |
| **Salida** | Bounding boxes | Decisión + razonamiento |
| **Velocidad** | ⚡ Muy rápido (50ms) | 🐢 Lento (2-3 min) |
| **Costos** | 💰 Local (GPU) | 💸 API Cloud |
| **Precisión** | Detección: 95%+ | Análisis: 90%+ |
| **Explicabilidad** | Baja | Muy alta |

---

### Flujo Integrado

```
Frame → YOLOv8 (¿Hay balón?) → SÍ → Captur 70 frames
                                  ↓
                            Mini-video
                                  ↓
                          Gemini (¿Fue canasta?)
                                  ↓
                        VideoResponse Final
```

---

## 🎯 SLIDE 4: RESULTADOS

### Título
**"Resultados del Sistema"** (Complétalo con tus datos)

### Sugerencias de Contenido

#### Sección A: Métricas de Desempeño
- Precisión de detección (%)
- Recall del modelo
- F1-Score
- Tiempo promedio de procesamiento

#### Sección B: Ejemplos de Casos
- Canasta encestada (caso exitoso)
- Canasta fallida (caso exitoso)
- Casos ambiguos (análisis del modelo)

#### Sección C: Análisis de Confianza
- Distribución de confianza en predicciones
- Ejemplos de alta/baja confianza

#### Sección D: Impacto
- Procesamiento de videos (cantidad, duración)
- Aplicaciones potenciales
- Mejoras futuras

---

## 📋 RECURSOS ADICIONALES

### Imágenes Sugeridas por Slide

**Slide 1 (Motivación):**
- Foto de jugador de baloncesto
- Icono de IA/Machine Learning
- Videos siendo procesados

**Slide 2 (Arquitectura):**
- Diagrama de flujo (incluido arriba)
- Logo FastAPI
- Logo PyTorch

**Slide 3 (Modelos):**
- Ejemplo de detección YOLOv8 (bounding boxes)
- Screenshot de Gemini API
- Comparación lado a lado

**Slide 4 (Resultados):**
- Gráficos de métricas
- Screenshots de interfaz web
- Videos de ejemplo procesados

---

## 🎨 TIPS DE DISEÑO PARA POWERPOINT

1. **Colores sugeridos:**
   - Naranja/Rojo (Canastas)
   - Azul (Tecnología)
   - Blanco (Fondo limpio)

2. **Tipografía:**
   - Títulos: Arial Bold, 54pt
   - Contenido: Calibri, 24pt
   - Datos: Monospace, 18pt

3. **Layout:**
   - Título arriba (centrado)
   - Contenido con viñetas
   - Máximo 5 líneas por slide
   - Imágenes ocupando 40-50% del espacio

4. **Transiciones:**
   - Simples (Fade, Wipe)
   - No distraer del contenido

---

**Nota:** Ajusta números, métricas e imágenes según tus resultados reales.
