import os
import uuid
import conf as conf
from logger import MainLogger, G29Logger

if __name__ == "__main__":
    # Create output directories
    output_dir = "logs/"
    # os.mkdir(output_dir)

    # Output locations
    logger = G29Logger(30, output_dir)
    logger.log()
