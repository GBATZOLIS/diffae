import os

def load_riemannian_config(path):
    """
    Loads a Python file that defines a CONFIG dict.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Riemannian config file not found: {path}")
    config_dict = {}
    with open(path, "r") as f:
        code = compile(f.read(), path, "exec")
        exec(code, config_dict)
    if "CONFIG" not in config_dict:
        raise ValueError(f"No 'CONFIG' dictionary found in {path}")
    return config_dict["CONFIG"]
