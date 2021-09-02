import logging
from logging.config import fileConfig

# import the configuration from logging_config
fileConfig('utils/logging_config.ini')
# instantiate our configurated logger
logger = logging.getLogger()