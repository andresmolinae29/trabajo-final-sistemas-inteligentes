PROMPT_TEMPLATE = """
    TAREA: Analizar trayectoria de baloncesto con MÁXIMA PRECISIÓN en cada frame.

    INSTRUCCIONES CRÍTICAS:
    1. Observa la TRAYECTORIA COMPLETA del balón en esta secuencia de video
    2. Identifica: posición inicial, arco de vuelo, punto final
    3. Busca el AMA (aro metálico) como punto de referencia
    4. Evalúa si el balón atraviesa el aro de ARRIBA → ABAJO
    5. Considera rebotes o salidas posteriores al contacto con el aro

    VALIDACIÓN ESTRICTA:
    - ¿El balón pasa DENTRO del aro (no solo cerca)?
    - ¿Va de arriba hacia abajo (no lateral ni inverso)?
    - ¿Hay rebote/salida después? (Si rebota afuera = NO ENCESTA)
    - Confianza en tu análisis (0-100%) (Si el porcentaje es menor a 70%, responde INDETERMINADO)

    RESPONDA ÚNICAMENTE EN JSON VÁLIDO:
    {
        "razonamiento": "<describe cada fase del movimiento>",
        "confianza_porcentaje": <0-100>,
        "resultado": "<ENCESTA|NO_ENCESTA|INDETERMINADO>",
    }
"""


class PromptProvider:
    BASKET_ANALYSIS = PROMPT_TEMPLATE
    @classmethod
    def get_prompt(cls, prompt_type="basket"):
        return getattr(cls, prompt_type.upper(), cls.BASKET_ANALYSIS)