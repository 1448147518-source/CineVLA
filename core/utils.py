"""Logger utility."""


def init_logger(filename):
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    fh = logging.FileHandler(filename, mode='w')
    fh.setLevel(logging.DEBUG); fh.setFormatter(fmt); logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG); ch.setFormatter(fmt); logger.addHandler(ch)
    return logger
