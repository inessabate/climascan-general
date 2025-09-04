import logging
import logging.config
import yaml
import os

def setup_logging(config_path="config/logging.yaml", default_level=logging.INFO):
    """
    Loads logging configuration from a YAML file.
    """

    # Find root directory of the project
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    abs_config_path = os.path.join(project_root, config_path)

    if os.path.exists(abs_config_path):
        with open(abs_config_path, "r") as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
        logging.getLogger().setLevel(default_level)
    else:
        logging.basicConfig(level=default_level)
        logging.warning(f"Logging config file not found: {abs_config_path}, using default basic config.")