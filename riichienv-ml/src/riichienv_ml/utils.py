import sys
from datetime import datetime
from pathlib import Path

from loguru import logger


def setup_logging(output_dir: str, script_name: str) -> Path:
    """Configure loguru to log to both stderr and a file.

    Log file is created in ``output_dir`` with name
    ``{script_name}_{YYYYMMDD_HHMMSS}.log``.

    Returns the log file path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"{script_name}_{timestamp}.log"

    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(str(log_path), level="DEBUG")
    logger.info(f"Logging to {log_path}")
    return log_path


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
