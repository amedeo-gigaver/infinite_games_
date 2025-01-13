# -- DO NOT TOUCH BELOW - ENV SET --
# flake8: noqa: E402
import os
import sys

# Force torch - must be set before importing bittensor
os.environ["USE_TORCH"] = "1"

# Add the parent directory of the script to PYTHONPATH
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)
# -- DO NOT TOUCH ABOVE --

import asyncio

from neurons.validator.main import main
from neurons.validator.utils.logger.logger import logger

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        exit(0)
    except Exception:
        logger.exception("Unexpected error")
        exit(1)
