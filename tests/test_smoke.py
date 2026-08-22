"""Smoke tests — verify the package is importable and environment is quiet."""

import os
import warnings


def test_import() -> None:
    import verascan  # noqa: F401


def test_version_string() -> None:
    from verascan import __version__

    assert isinstance(__version__, str)
    assert __version__ == "0.1.0"


def test_quiet_environment_configured() -> None:
    """Verify that Verascan sets up quiet environment variables for ML deps."""
    import verascan._env  # noqa: F401

    assert os.environ.get("TF_ENABLE_ONEDNN_OPTS") == "0"
    assert os.environ.get("TF_CPP_MIN_LOG_LEVEL") == "3"
    assert os.environ.get("HF_HUB_DISABLE_SYMLINKS_WARNING") == "1"


def test_clean_import_no_tensorflow_warnings() -> None:
    """Verify importing verascan does not trigger unhandled deprecation warnings."""
    with warnings.catch_warnings(record=True) as recorded:
        import verascan  # noqa: F401

        tf_warnings = [
            w
            for w in recorded
            if "tensorflow" in str(w.message).lower() or "tf_keras" in str(w.message).lower()
        ]
        assert len(tf_warnings) == 0
