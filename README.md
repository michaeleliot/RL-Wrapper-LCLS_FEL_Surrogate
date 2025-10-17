# LCLS_FEL_Surrogate
Self-contained ML-based surrogate model of the LCLS FEL pulse intensity packaged using LUME-model [text](https://slaclab.github.io/lume-model/).

## Dependencies
```
lume-model
```

## Usage
From the main repoistory directory, call
```python
from lume-model.models import TorchModel

# load model from yaml
model = TorchModel("model_config.yaml")

# evaluate the model at a given point
model.evaluate({"ACCL:LI25:1:ADES": 6260.0})
```
 NOTE: when not specified, input variables are set to their default values
 as defined in `model_config.yaml`