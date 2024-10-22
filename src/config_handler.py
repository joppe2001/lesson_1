import tomli
import os
from pathlib import Path

class ConfigHandler:
    def __init__(self, config_filename="config.toml"):
        self.base_dir = Path(__file__).parent.parent  # Goes up from src to root
        self.config_path = self.base_dir / config_filename
        self.config = self._load_config()

    def _load_config(self):
        """Load the TOML configuration file."""
        try:
            with open(self.config_path, "rb") as f:
                return tomli.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Configuration file not found at {self.config_path}. "
                f"Current working directory: {os.getcwd()}, "
                f"Base directory: {self.base_dir}"
            )

    def get_processed_file_path(self):
        """Get the full path to the current processed data file."""
        return self.base_dir / self.config["processed"] / self.config["current"]

    def get_raw_file_path(self):
        """Get the full path to the raw input file."""
        return self.base_dir / self.config["raw"] / self.config["input"]

    def get_image_dir(self):
        """Get the path to the Images directory."""
        return self.base_dir / "images"

    def ensure_directories(self):
        """Ensure all necessary directories exist."""
        directories = [
            self.base_dir / self.config["raw"],
            self.base_dir / self.config["processed"],
            self.get_image_dir()
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def __str__(self):
        """String representation for debugging."""
        return (f"ConfigHandler:\n"
                f"  Base Directory: {self.base_dir}\n"
                f"  Config Path: {self.config_path}\n"
                f"  Config Contents: {self.config}")