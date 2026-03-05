import cv2
import os

from ultralytics import YOLO
from google import genai
from PIL import Image


model_yolo = YOLO("yolo26n.pt")
GEMINI_API_KEY = os.getenv("API_KEY_GEMINI")
client = genai.Client(api_key="")


def analize_frame_with_llm(frame):

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    response = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=[
            img,
            "El balon de baloncesto esta entrando al ara o acaba de encestar"
            "Responde solo: ENCESTA, NO ENCESTA o INDETERMINADOs"
            ]
    )

    return response.text.strip()


def process_video(video_path = 0):

    cap = cv2.VideoCapture(video_path)

    decisions = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model_yolo(frame)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                class_id = int(box.cls[0])

                if class_id == 32 and confidence > 0.5:  # Clase 32 es balon de baloncesto
                    cropped_frame = frame[y1:y2, x1:x2]
                    # decision = analize_frame_with_llm(cropped_frame)
                    decision = "ENCESTA"  # Simulación de decisión para pruebas
                    decisions.append(decision)

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return decisions

if __name__ == "__main__":
    result = process_video("tests/test_videos/first-video.mp4")
    print(result)
