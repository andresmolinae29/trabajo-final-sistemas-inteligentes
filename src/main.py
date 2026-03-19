import shutil

import uvicorn
from pathlib import Path
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from basketball_detector.ai_models import GoogleVideoModelFactory
from basketball_detector.services import DetectionServiceFactory, VideoService
from basketball_detector.utils import TempFileVideosManager, BASE_DIR, logger


app = FastAPI()


app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

results_store = {}


@app.get("/")
def home(request: Request, result_id: str | None = None):
    result = results_store.get(result_id) if result_id else None
    return templates.TemplateResponse(
        "home.html", {"request": request, "result": result}
    )


@app.post("/upload")
def upload_video(request: Request, file: UploadFile = File(...)):
    temp_video_manager = TempFileVideosManager()
    video_path = temp_video_manager.original_video_path
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    basketball_service = DetectionServiceFactory.create()
    google_video_model = GoogleVideoModelFactory.create()
    video_service = VideoService(basketball_service, google_video_model)
    decisions = video_service.process_video(str(video_path))

    logger.info(f"Decisiones obtenidas: {decisions}")

    mini_videos = []
    for _, decision in enumerate(decisions):

        mini_videos.append(
            {
                "path": f"/static/tmp/{decision.video_name}",
                "result": decision.result,
                "confidence_percentage": decision.confidence_percentage,
                "reasoning": decision.reasoning,
            }
        )

    results_store[temp_video_manager.temp_video_id] = {
        "original": f"/static/tmp/{temp_video_manager.original_file}",
        "mini_videos": mini_videos,
    }
    return RedirectResponse(
        url=f"/?result_id={temp_video_manager.temp_video_id}", status_code=303
    )


if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True, workers=2)
