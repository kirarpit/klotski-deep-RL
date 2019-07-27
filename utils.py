import logging
import os
import platform


def setup_logger(name, level=logging.INFO):
    log_dir = "logs/"
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    handler = logging.FileHandler(log_dir + name + "-" + platform.node() + '.log')
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(handler)
    return logger
