"""Common utilities."""

import logging
import sys
from dataclasses import dataclass, field
from os.path import basename, splitext
import numpy as np


@dataclass
class Score:
    """Score for the """
    train_losses: list[float] = field(default_factory=list)
    score_losses: list[float] = field(default_factory=list)
    total: list[float] = field(default_factory=list)
    correct_1: list[float] = field(default_factory=list)
    avg_train_loss: float = float('infinity')
    avg_score_loss: float = float('infinity')
    top_1: float = 0.0
    top_k: float = 0.0

    def calc(self):
        """Calculate the scores based on raw data."""
        self.avg_train_loss = np.mean(self.train_losses)
        self.avg_score_loss = np.mean(self.score_losses)
        if self.total:
            self.top_1 = 100.0 * sum(self.correct_1) / sum(self.total)

    def better_than(self, other):
        """Check if this score is better than the other one."""
        self.calc()
        return (-self.top_1, self.avg_score_loss) < (-other.top_1, other.avg_score_loss)


def setup_logger(level=logging.INFO):
    """Setup the logger."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')


def module_name() -> str:
    """Get the current module name."""
    return splitext(basename(sys.argv[0]))[0]


def started() -> None:
    """Log the program start time."""
    setup_logger()
    logging.info('=' * 80)
    logging.info(f'{module_name()} started')


def finished() -> None:
    """Log the program end time."""
    logging.info(f'{module_name()} finished')
