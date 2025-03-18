import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import trunc_normal_, Mlp, PatchEmbed
from xformers.ops import memory_efficient_attention, unbind
from functools import lru_cache

import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), "../"))
from model.utils import get_2d_sincos_pos_embed, get_1d_sincos_pos_embed_from_grid, get_pad2d


class OceanvariableEmbedding(nn.Module):
    def __init__(
        self,
        variables,
        img_size,
        patch_size=2,
        embed_dim=1024,
        num_heads=16,
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.variables = variables

        # variable tokenization: separate embedding layer for each input variable
        self.token_embeds = nn.ModuleList(
            [PatchEmbed(None, patch_size, 1, embed_dim) for i in range(len(variables))]
        )
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)

        # variable embedding to denote which variable each token belongs to
        # helps in aggregating variables
        self.channel_embed = self.create_var_embedding(embed_dim)

        # variable aggregation: a learnable query and a single-layer cross attention
        self.channel_query = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        self.channel_agg = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=True)

        self.initialize_weights()

    def initialize_weights(self):
        # initialize pos_emb and var_emb
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.img_size[0] / self.patch_size),
            int(self.img_size[1] / self.patch_size),
            cls_token=False,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        channel_embed = get_1d_sincos_pos_embed_from_grid(self.channel_embed.shape[-1], np.arange(self.variables))
        self.channel_embed.data.copy_(torch.from_numpy(channel_embed).float().unsqueeze(0))

        # token embedding layer
        for i in range(len(self.token_embeds)):
            w = self.token_embeds[i].proj.weight.data
            trunc_normal_(w.view([w.shape[0], -1]), std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def create_var_embedding(self, dim):
        var_embed = nn.Parameter(torch.zeros(1, self.variables, dim), requires_grad=True)
        return var_embed

    def get_var_emb(self):
        return self.channel_embed

    def aggregate_variables(self, x: torch.Tensor):
        """
        x: B, V, L, D
        """
        b, _, l, _ = x.shape
        x = torch.einsum("bvld->blvd", x)
        x = x.flatten(0, 1)  # BxL, V, D

        var_query = self.channel_query.repeat_interleave(x.shape[0], dim=0)
        x, _ = self.channel_agg(var_query, x, x)  # BxL, D
        x = x.squeeze()

        x = x.unflatten(dim=0, sizes=(b, l))  # B, L, D
        return x

    def forward(self, x: torch.Tensor):

        # tokenize each variable separately
        embeds = []

        for i in range(self.variables):
            embed_variable = self.token_embeds[i](x[:, i : i + 1]) # B, L, D
            embeds.append(embed_variable)
        x = torch.stack(embeds, dim=1)  # B, V, L, D

        # add variable embedding
        var_embed = self.get_var_emb()
        x = x + var_embed.unsqueeze(2)
        x = x + self.pos_embed.unsqueeze(1)

        # variable aggregation
        x = self.aggregate_variables(x)  # B, L, D

        return x

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.mlp = nn.Linear(2, hidden_size)

    def forward(self, t):
        return self.mlp(t)


class MemEffAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_bias=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """
    An transformers block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = MemEffAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.Identity()
        # self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class Xuanming(nn.Module):
    def __init__(self, 
        in_img_size=(400,441),
        variables=19,
        level_reduction=3,
        patch_size=2,
        hidden_size=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
    ):
        super().__init__()
        
        self.pad_size_h = 0
        self.pad_size_w = 0

        if in_img_size[0] % patch_size != 0 or in_img_size[1] % patch_size != 0:
            self.padding = True
            self.pad_size_h = patch_size - in_img_size[0] % patch_size
            self.pad_size_w = patch_size - in_img_size[1] % patch_size
            in_img_size = (in_img_size[0] + self.pad_size_h, in_img_size[1] + self.pad_size_w)

        self.in_img_size = in_img_size
        self.variables = variables
        self.patch_size = patch_size
        self.level_reduction = level_reduction
        
        # embedding
        self.embedding = OceanvariableEmbedding(
            variables=variables,
            img_size=in_img_size,
            patch_size=patch_size,
            embed_dim=hidden_size,
            num_heads=num_heads,
        )
        self.embed_norm_layer = nn.LayerNorm(hidden_size)
        
        # interval embedding
        self.t_embedder = TimestepEmbedder(hidden_size)
        
        # backbone
        self.blocks = nn.ModuleList([
            Block(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        
        # prediction layer
        self.head = FinalLayer(hidden_size, patch_size, variables)

        self.initialize_weights()

    def pad(self, x):
        padded_x = torch.nn.functional.pad(x, (self.pad_size_w, 0, self.pad_size_h, 0), 'constant', 0)
        return padded_x

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        trunc_normal_(self.t_embedder.mlp.weight, std=0.02)
        
        # Zero-out adaLN modulation layers in blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.head.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.head.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.head.linear.weight, 0)
        nn.init.constant_(self.head.linear.bias, 0)

    def unpatchify(self, x: torch.Tensor, h=None, w=None):
        """
        x: (B, L, V * patch_size**2)
        return imgs: (B, V, H, W)
        """
        p = self.patch_size
        v = len(self.variables)
        h = self.in_img_size[0] // p if h is None else h // p
        w = self.in_img_size[1] // p if w is None else w // p
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, v))
        x = torch.einsum("nhwpqv->nvhpwq", x)
        imgs = x.reshape(shape=(x.shape[0], v, h * p, w * p))
        return imgs

    def forward(self, x, x_mark, y_mark):
        B, C, N, H, W = x.shape
        x = x.reshpae(B, C, N//self.level_reduction, self.level_reduction, H, W)
        x = x.mean(dim=3)
        x = x.reshape(B, -1, H, W)

        if self.padding:
            x = self.pad(x)
        
        x = self.embedding(x) # B, L, D
        x = self.embed_norm_layer(x)

        x_time_emb = self.t_embedder(x_mark)
        for block in self.blocks:
            x = block(x, x_time_emb)
        
        y_time_emb = self.t_embedder(y_mark)
        x = self.head(x, y_time_emb)
        x = self.unpatchify(x)
        
        x = x[:,:,self.pad_size_h:, self.pad_size_w:]
        x = F.interpolate(x, size=(N, H, W), mode="trilinear")
        return x

if __name__=="__main__":
    model = Xuanming()
    x = torch.randn(2, 19, 30, 400, 441)
    x_mark = torch.Tensor([[11, 30], [1,28]]).long()
    x_mark = torch.Tensor([[11, 30], [1,28]]).long()
    y = model(x, x_mark, y_mark)