import logging
from rich.logging import RichHandler

def setup_logger(logger_name=__name__):
    # remove all handler from root
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logger = logging.getLogger(logger_name)

    # remove all handler from logger
    logger.handlers.clear()

    # log level
    logger.setLevel(logging.INFO)
    # logger.setLevel(logging.DEBUG)

    # handler
    handler = RichHandler(
        markup=True, 
        rich_tracebacks=True, 
        show_time=True, 
        show_level=True,
        show_path=True
        )

    # formatter
    formatter = logging.Formatter(
        fmt="%(message)s", 
        datefmt="[%H:%M:%S]"
        )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger

