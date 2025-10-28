import yaml

# Path to your configuration file
CONFIG_PATH = "model_config.yaml"

def extract_quad_ltuh_pvs(config_path: str):
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


QUAD_NAMES, RANGES, DEFAULTS = extract_quad_ltuh_pvs(CONFIG_PATH)