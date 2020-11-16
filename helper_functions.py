import os
import yaml
from addict import Dict


kwargs_creator = lambda x: x if x else {}

def load_cfg(yaml_filepath):
    """
    Load a YAML configuration file.
    """
    # Read YAML experiment definition file
    with open(yaml_filepath, 'r') as stream:
        cfg = yaml.load(stream, yaml.FullLoader)
    cfg = make_paths_absolute(os.path.dirname(yaml_filepath), cfg)
    return Dict(cfg)

def make_paths_absolute(dir_, cfg):
    """
    Make all values for keys ending with `_path` & `_file` absolute to dir_.
    """
    for key in cfg.keys():
        if key.endswith("_path") or key.endswith("_file"):
            cfg[key] = os.path.join(dir_, cfg[key])
            cfg[key] = os.path.abspath(cfg[key])
        if isinstance(cfg[key], dict):
            cfg[key] = make_paths_absolute(dir_, cfg[key])
    return cfg

def create_folder(folder, rm_existing=False):
    exists = os.path.exists(folder)
    if rm_existing and exists:
        os.rmdir(folder)
    os.makedirs(folder, exist_ok=True)
