import logging
from pathlib import Path

from datetime import datetime

logs_dir = Path(__file__).parent.parent.parent.parent / "logs"
logs_dir.mkdir(exist_ok=True)

log_file = logs_dir / f"basketball_detector_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
logging.basicConfig(
    encoding='utf-8',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
