import logging
from logging import Formatter, Logger, LogRecord, getLogger
from types import MethodType

from numpy.typing import NDArray


class Color:
    GREEN = '\33[1;32m'
    YELLOW = '\33[1;33m'
    NONE = '\33[0m'


class CustomFormatter(Formatter):
    def __init__(self, format: str) -> None:
        super().__init__(format)

    def format(self, record: LogRecord) -> str:
        original_format = self._style._fmt
        match record.levelno:
            case logging.INFO:
                self._style._fmt = f'{Color.GREEN}{original_format}{Color.NONE}'
            case logging.WARNING | logging.ERROR | logging.CRITICAL:
                self._style._fmt = f'{Color.YELLOW}{original_format}{Color.NONE}'
        result = super().format(record)
        self._style._fmt = original_format
        return result


def getCustomLogger(name: str | None = None) -> Logger:
    def _new_log(logger: Logger, level, msg, args, **kw) -> None:
        if isinstance(msg, str):
            for sub_msg in msg.split('\n'):
                getattr(logger, '_original_log')(level, sub_msg, args, **kw)
        else:
            getattr(logger, '_original_log')(level, msg, args, **kw)

    logger = getLogger(name)
    setattr(logger, '_original_log', MethodType(Logger._log, logger))
    setattr(logger, '_log', MethodType(_new_log, logger))
    return logger


logger = getCustomLogger()


def set_verbosity() -> None:
    plt_logger = getCustomLogger('matplotlib.font_manager')
    plt_logger.setLevel(logging.INFO)


def debug(count: int = 1):
    def _debug(*args: NDArray) -> None:
        nonlocal count
        for array in args:
            logger.info(array.shape)
        count -= 1
        if count <= 0:
            exit(0)

    return _debug
