import logging
import logging.config


def make_logger(name=None):
    """ setup logging """
    if name:
        logger = logging.getLogger(name)
    else:
        logger = logging.getLogger(__name__)

    fmt = '%(asctime)s [%(levelname)s]%(name)s: %(message)s'

    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,

        'formatters': {'standard': {'format': fmt,
                                    'datefmt': '%Y-%m-%d %H:%M:%S'}},

        'handlers': {'console': {'level': 'INFO',
                                 'class': 'logging.StreamHandler',
                                 'formatter': 'standard'},

                     'debug_file': {'class': 'logging.FileHandler',
                                    'level': 'DEBUG',
                                    'filename': './debug.log',
                                    'mode': 'w',
                                    'formatter': 'standard'},

                     'info_file': {'class': 'logging.FileHandler',
                                   'level': 'INFO',
                                   'filename': './info.log',
                                   'mode': 'w',
                                   'formatter': 'standard'}},

        'loggers': {'': {'handlers': ['console', 'debug_file', 'info_file', ],
                         'level': 'DEBUG',
                         'propagate': True}}})

    return logger
