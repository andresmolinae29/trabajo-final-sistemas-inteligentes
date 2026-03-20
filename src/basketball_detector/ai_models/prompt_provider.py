PROMPT_TEMPLATE = """
    Eres un árbitro experto de baloncesto. Analiza este video frame a frame para determinar si el balón entra en la canasta.

    ANÁLISIS REQUERIDO (en orden):
    1. LOCALIZA el aro y el balón en cada frame clave
    2. TRAZA la trayectoria completa: lanzamiento → vuelo → contacto con aro/tablero → resultado
    3. DETERMINA si el balón atraviesa el aro de ARRIBA hacia ABAJO completamente
    4. OBSERVA la red: ¿se mueve hacia abajo indicando que el balón pasó a través?

    CRITERIOS DE DECISIÓN:
    - ENCESTA: El balón pasa completamente a través del aro (de arriba a abajo) y/o la red se agita hacia abajo
    - NO_ENCESTA: El balón rebota fuera del aro, pasa por fuera, golpea tablero sin entrar, o no llega al aro
    - INDETERMINADO: Vista obstruida, balón fuera de cuadro en momento clave, o confianza < 70%

    RESPONDE SOLO con este JSON (sin texto adicional):
    {
        "razonamiento": "<describe trayectoria fase por fase: lanzamiento, vuelo, contacto, resultado>",
        "confianza_porcentaje": <0-100>,
        "resultado": "<ENCESTA|NO_ENCESTA|INDETERMINADO>"
    }
"""


class PromptProvider:
    BASKET_ANALYSIS = PROMPT_TEMPLATE
    @classmethod
    def get_prompt(cls, prompt_type="basket"):
        return getattr(cls, prompt_type.upper(), cls.BASKET_ANALYSIS)