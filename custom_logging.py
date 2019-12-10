import logging
import colorlog

LL = logging.INFO
LOGFORMAT = "  %(log_color)s%(levelname)-8s%(reset)s | %(log_color)s%(module)s%(reset)s | %(log_color)s%(message)s%(reset)s"
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(LOGFORMAT))


def get_logger(name):
    logger = colorlog.getLogger(name)
    logger.setLevel(LL)
    logger.propagate = False
    if not len(logger.handlers):
        logger.addHandler(handler)

    return logger