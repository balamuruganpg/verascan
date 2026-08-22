"""Internal environment and warning configurations for Verascan.

Ensures that optional heavy dependencies (TensorFlow, Keras, transformers,
huggingface_hub) operate quietly without spamming console stderr with oneDNN,
deprecation, and cache symlink warnings during normal operation.
"""

from __future__ import annotations

import logging
import os
import warnings


def configure_quiet_environment() -> None:
    """Configure environment variables and warning filters to silence noisy logs.

    Uses `setdefault` so any existing environment variables set by the user
    are preserved.
    """
    # 1. TensorFlow / oneDNN C++ logging levels
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    # 2. Hugging Face & Transformers noisy warnings
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # 3. Suppress known deprecation / cosmetic warnings from TF/Keras shims
    warnings.filterwarnings("ignore", category=UserWarning, module=r"tensorflow.*")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"tensorflow.*")
    warnings.filterwarnings("ignore", category=UserWarning, module=r"tf_keras.*")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"tf_keras.*")
    warnings.filterwarnings("ignore", category=UserWarning, module=r"huggingface_hub.*")

    # 4. Logger noise reduction
    for name in ("tensorflow", "transformers", "huggingface_hub", "urllib3"):
        logging.getLogger(name).setLevel(logging.ERROR)


# Execute immediately on module import so any subsequent imports are quiet.
configure_quiet_environment()
