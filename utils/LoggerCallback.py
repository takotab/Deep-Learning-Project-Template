import time
import logging

from fastai.torch_core import ifnone
from fastai.metrics import accuracy
from fastai.basic_train import LearnerCallback, Learner


class LoggingLog(LearnerCallback):
    "A `LearnerCallback` that logs metrics while training `learn`."

    def __init__(self, learn: Learner, loggername: str, timer: bool = True):
        super().__init__(learn)

        self.logger = logging.getLogger(loggername)
        self.timer = Timer() if timer else None

    def on_train_begin(self, **kwargs) -> None:
        "Prepare file with metric names."
        self.logger.info("Start training")
        if self.timer:
            self.timer.start()

    def on_epoch_end(self, epoch: int, smooth_loss, last_metrics, **kwargs) -> bool:
        "Add a line with `epoch` number, `smooth_loss` and `last_metrics`."
        if last_metrics is None:
            last_metrics = []
        if self.timer:
            timer_info = self.timer.lap(epoch)
        stats = [
            str(stat)
            if isinstance(stat, int)
            else "#na#"
            if stat is None
            else f"{stat:.3f}"
            for stat in [epoch, smooth_loss] + last_metrics + timer_info
        ]
        str_stats = "\t".join(stats)
        self.logger.info(str_stats + "\n")


class Timer:
    def __init__(self):
        self.start_time = time.time()
        self.laps = [0]

    def start(self):
        self.start_time = time.time()

    def lap(self, epoch):
        elapsed_time = time.time() - self.start_time
        self.laps.append(elapsed_time)
        return [self.laps[-1] - self.laps[-2], self.laps[-1]]
