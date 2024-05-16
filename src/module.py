import torch
import torch.nn as nn
from einops import rearrange
from timm.models.vision_transformer import Mlp


def exists(x):
    return x is not None


class Attention(nn.Module):
    """
    reference
    [1] https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
    [2] https://github.com/lucidrains/imagen-pytorch/blob/main/imagen_pytorch/imagen_pytorch.py
    """
    def __init__(self, dim, heads = 8, dropout = 0.):
        super().__init__()
        assert dim % heads == 0
        dim_head = dim // heads

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)

    def forward(self, x, mask = None):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        sim = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if exists(mask):
            # 0 -> not masked, 1 -> masked
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(mask > 0.5, max_neg_value)

        sim = sim.softmax(dim = -1)
        sim = self.drop(sim)

        out = torch.matmul(sim, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return out
    

class Block(nn.Module):
    """
    A block with adaptive layer norm zero (adaLN-Zero) conditioning and attention masks.
    reference
    [1] https://github.com/facebookresearch/DiT/blob/main/models.py
    """
    def __init__(self, dim, heads = 8, mlp_ratio = 4.0, dropout = 0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine = False, eps = 1e-6)
        self.attn = Attention(dim, heads = heads, dropout = dropout)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine = False, eps = 1e-6)

        mlp_hidden_dim = int(dim * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate = 'tanh')
        self.mlp = Mlp(in_features = dim, hidden_features = mlp_hidden_dim, act_layer = approx_gelu, drop = dropout)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias = True)
        )

    @staticmethod
    def rearrange(x):
        return rearrange(x, 'b d -> b 1 d')

    def forward(self, x, mask = None, cond = None):
        if exists(cond):
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(cond).chunk(6, dim = 1)
            shift_msa = self.rearrange(shift_msa)
            scale_msa = self.rearrange(scale_msa)
            gate_msa = self.rearrange(gate_msa)
            shift_mlp = self.rearrange(shift_mlp)
            scale_mlp = self.rearrange(scale_mlp)
            gate_mlp = self.rearrange(gate_mlp)

            x = x + gate_msa * self.attn(self.norm1(x) * (1 + scale_msa) + shift_msa, mask = mask)
            x = x + gate_mlp * self.mlp(self.norm2(x) * (1 + scale_mlp) + shift_mlp)
            return x
                
        else:
            x = x + self.attn(self.norm1(x), mask = mask)
            x = x + self.mlp(self.norm2(x))
            return x
        

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads = 8, mlp_ratio = 4.0, dropout = 0.):
        super().__init__()

        self.blocks = nn.ModuleList([
            Block(dim, heads = heads, mlp_ratio = mlp_ratio, dropout = dropout) for _ in range(depth)
        ])

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Zero-out adaLN modulation layers in blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def forward(self, x, mask = None, cond = None):
        for block in self.blocks:
            x = block(x, mask = mask, cond = cond)
        return x