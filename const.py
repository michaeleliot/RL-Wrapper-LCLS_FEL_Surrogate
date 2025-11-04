import yaml
import numpy as np
# Import your environment class
# from ltuh_env import LTUHEnv 

# Path to your configuration file
CONFIG_PATH = "model_config.yaml"

def extract_quad_ltuh_pvs(config_path: str):
    """
    Extracts quad names, value ranges, and defaults from a YAML config.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    input_vars = config.get("input_variables", {})

    quad_ltuh_names = []
    value_range_map = {}
    default_value_map = {}

    for name, attrs in input_vars.items():
        if name.startswith("QUAD:LTUH"):
            quad_ltuh_names.append(name)
            value_range_map[name] = attrs.get("value_range", None)
            default_value_map[name] = attrs.get("default_value", None)

    return quad_ltuh_names, value_range_map, default_value_map


# --- 1. Extract values from config ---
# These RANGES represent the *model's hard limits*
QUAD_NAMES, RANGES, DEFAULTS = extract_quad_ltuh_pvs(CONFIG_PATH)

# --- 2. Define your buffer and create the tightened ranges ---
BUFFER = 1  # A small, safe margin
buffered_env_ranges = {}

for name in QUAD_NAMES:
    value_range = RANGES.get(name)
    
    if value_range is None or len(value_range) != 2:
        print(f"Warning: No valid 'value_range' for {name}. Skipping.")
        continue

    # Get the model's true hard limits
    model_min = float(value_range[0])
    model_max = float(value_range[1])

    # Apply the buffer to create a tighter range for the env
    env_min = model_min + BUFFER
    env_max = model_max - BUFFER

    # Sanity check: Ensure buffer isn't too large
    if env_min >= env_max:
        raise ValueError(
            f"Buffer of {BUFFER} is too large for quad '{name}' "
            f"with original range [{model_min}, {model_max}]"
        )
        
    buffered_env_ranges[name] = [env_min, env_max]
