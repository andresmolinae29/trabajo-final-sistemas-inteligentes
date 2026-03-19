import logging
from pathlib import Path

logs_dir = Path(__file__).parent.parent.parent.parent / "logs"
logs_dir.mkdir(exist_ok=True)

log_file = logs_dir / "basketball_detector.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
