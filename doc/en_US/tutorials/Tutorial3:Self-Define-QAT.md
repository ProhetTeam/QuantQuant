## __Turorial 3 : Self-Define Quant Transformer__

To customize a new QAT approach, there are the following steps:

-  Define a new quantization class
-  Import the module
-  Modify the quantization setting in config file

### __Define a new quantization class__

Suppose we want to devolop a new quantization method DSQv2. First, create a new folder ```DSQv2``` under ```QuanTransformer/quantrans/quantops'```, and create a new file ```QuanTransformer/quantrans/quantops/DSQv2/DSQConvV2.py'```.


```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import QUANLAYERS

@QUANLAYERS.register_module()
class DSQConvV2(nn.Conv2d):
    def __init__(self, *args):

    def quantize_(self, x, *args):
        pass

    def forward(self, x):
        pass

```

Note that after defining the class, we add a decorator function: 
```@QUANLAYERS.register_module()```.
Then,  we import this module in ```QuanTransformer/quantrans/quantops/DSQv2/__init__.py'```: 


```python
from .DSQConvV2 import DSQConvV2

__all__=['DSQConvV2']
```

this process is equivalent to excuting  ```DSQConvV2 = QUANLAYERS.register_module(DSQConvV2()) ```.

### __Modify the quantization setting in config file__

```python
quant_transformer = dict(
    type = "mTransformerV2",
    quan_policy=dict(
        Conv2d=dict(type='DSQConvV2', num_bit_w=3, num_bit_a=3, bSetQ=True),
        ),
    special_layers = dict(
        layers_name = [
            'backbone.conv1',
            'head.fc'],
        convert_type = [dict(type='DSQConv', num_bit_w=8, num_bit_a=8, bSetQ=True, quant_activation=False),
                        dict(type='DSQLinear', num_bit_w=8, num_bit_a=8)]
        )
)
```


### __How the new modules are used to quantify the model__

Using quantization setting as a argument, we define a model transformer:
```python
model_transformer = build_mtransformer(cfg.quant_transformer)
```

In ```__init__``` function, the dict of quantization setting is assigned to ```self.register_dict```:
```python
class mTransformerV2(Basemtransformer, nn.Module):
    def __init__(self, 
                 quan_policy = dict(),
                 special_layers = None,
                 **kwargs):
        super(mTransformerV2, self).__init__()
        self.special_layers = special_layers

        self.register_dict = OrderedDict()
        for key, value in quan_policy.items():
            assert(hasattr(nn, key))
            self.register_dict[getattr(nn, key)] = value
        self.layer_idx = 0
```

Then, the transformer convert a standard model into a quantization model:

```python
model = model_transformer(model, logger = logger)
```

For each module, the transformer first gets the name and check whether it exists in ```self.register_dict```:

```python
if type(current_layer) not in self.register_dict:
    continue
```

If the current layer can be converted, first extract the parameters of the original layer ```new_kwargs```, 

```python
## 1. get parameters
sig = inspect.signature(type(getattr(model, module_name)))
new_kwargs = {}
for key in sig.parameters:
    if sig.parameters[key].default != inspect.Parameter.empty:
        continue
    assert(hasattr(current_layer, key))
    new_kwargs[key] = getattr(current_layer, key)
```

and get the corresponding quantization arguments ```quan_args``` according to the layer name.
These arguments are combined to build a new quant layer. The weights of current layer is merged to quant layer, so the quant layer can use it for operations like convolution. Finally, the quantization layer replaces the current layer in the model.

```python
## 2. Special layers or Normal layer
if current_layer_name in self.special_layers.layers_name:
    idx = self.special_layers.layers_name.index(current_layer_name)
    quan_args = self.special_layers.convert_type[idx]
else:
    quan_args = self.register_dict[type(current_layer)]
    new_kwargs = {**quan_args, **new_kwargs}
    new_quan_layer = build_quanlayer(new_kwargs)
    dict_merge(new_quan_layer.__dict__, current_layer.__dict__)
    setattr(model, module_name, new_quan_layer)
```
