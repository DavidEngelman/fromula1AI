import os
import uuid
import src.conf as conf
from src.logger import MainLogger

if __name__ == "__main__":
    # Create output directories
    output_dir = "logs/"
    os.mkdir(output_dir)

    # Output locations
    logger = MainLogger(30, output_dir)
    logger.log()
