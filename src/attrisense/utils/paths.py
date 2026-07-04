"""Project path resolution.

Locates the repository root by walking up from the current file until
``configs/config.yaml`` is found, then exposes standard directory constants
used across the package and Streamlit app.
"""

from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """Walk up from *start* until a directory containing ``configs/config.yaml`` is found.

    Args:
        start: Directory to begin the search from. Defaults to this file's location.

    Returns:
        Absolute path to the repository root.

    Raises:
        FileNotFoundError: If no project root marker is found.
    """
    current = (start or Path(__file__)).resolve()
    if current.is_file():
        current = current.parent

    for candidate in [current, *current.parents]:
        if (candidate / "configs" / "config.yaml").exists():
            return candidate

    raise FileNotFoundError(
        "Could not locate project root (expected configs/config.yaml in an ancestor directory)."
    )


PROJECT_ROOT = find_project_root()
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
