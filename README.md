# Masked Conditional Transformer

The Masked Conditional Transformer is a neural network architecture designed to transform data based on specific conditions. This model employs attention mechanisms and conditional normalization to learn and predict intricate patterns in input data.

### Features
1. Attention Mask: Masking unwanted data.
2. Adaptive Normalization (adaLN-Zero): Changing data based on conditions.

### Example
```python
import torch
from src.module import Transformer

# define transformer
transformer = Transformer(
    dim = 4, 
    depth = 3,
    heads = 1,
    mlp_ratio = 4,
    dropout = 0.1
)

# inputs
x = torch.randn(1, 3, 4)

# mask
mask = torch.zeros(1, 3)
mask[:, 2:] = 1

# condition
cond = torch.randn(1, 4)

```

```python
transformer(x, mask = mask, cond = cond)
>>> tensor([[[-0.5015,  0.7465, -0.4192, -1.2293],
             [-0.4267, -0.9740, -0.3389, -1.6674],
             [-0.3586,  0.3017,  1.3145, -0.9639]]], grad_fn=<AddBackward0>)
```

### Reference
1. https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
2. https://github.com/lucidrains/imagen-pytorch/blob/main/imagen_pytorch/imagen_pytorch.py 
3. https://github.com/facebookresearch/DiT/blob/main/models.py