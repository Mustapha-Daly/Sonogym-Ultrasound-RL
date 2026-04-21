"""
Python module serving as a project/extension template.
"""

# Register Gym environments.
import os

from .lab import *
from .tasks import *

# Register UI extensions.
#from .ui_extension_example import *


ASSETS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "assets"))

ASSETS_DATA_DIR = os.path.abspath(os.path.join(ASSETS_EXT_DIR, "data"))

PACKAGE_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

# Register UI extensions.
def _try_register_ui():
    try:
        import omni.ext  # only exists after SimulationApp
        import spinal_surgery.ui_extension_example
    except Exception:
        # Safe to ignore: headless mode, registry scan, CLI, tests
        pass

_try_register_ui()

