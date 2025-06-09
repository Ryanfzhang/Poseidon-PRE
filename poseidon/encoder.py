import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import trunc_normal_, Mlp, PatchEmbed
from functools import lru_cache
from einops import rearrange
from typing import Optional

import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), "../"))
from poseidon.attention import PerceiverResampler
from poseidon.utils import get_2d_sincos_pos_embed, get_1d_sincos_pos_embed_from_grid, get_pad2d


class Encoder(nn.Module):
    def __init__(
        self,
        img_size,
        variables=19,
        latent_levels=8,
        levels=30,
        patch_size=2,
        embed_dim=1024,
        num_heads=16,
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.variables = variables

        self.token_embeds = nn.ModuleList([PatchEmbed(None, patch_size, variables, embed_dim) for i in range(levels)])
        self.levels = levels
        self.latent_levels = latent_levels
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)

        # variable aggregation: a learnable query and a single-layer cross attention
        self.level_latents = nn.Parameter(torch.randn(latent_levels, embed_dim))
        self.level_agg = PerceiverResampler(
            latent_dim=embed_dim,
            context_dim=embed_dim,
            depth=2,
            head_dim=64,
            num_heads=num_heads,
            drop=0.1,
            mlp_ratio=4.0,
            ln_eps=1e-5,
            ln_k_q=False,
        )

        # positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=True)
        self.dropout = nn.Dropout(0.1)
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

        # token embedding layer
        w = self.token_embeds.proj.weight.data
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

    def aggregate_levels(self, x: torch.Tensor) -> torch.Tensor:
        """Aggregate pressure level information.

        Args:
            x (torch.Tensor): Tensor of shape `(B, C_A, L, D)` where `C_A` refers to the number
                of pressure levels.

        Returns:
            torch.Tensor: Tensor of shape `(B, C, L, D)` where `C` is the number of
                aggregated pressure levels.
        """
        B, _, L, _ = x.shape
        latents = self.level_latents
        latents = latents.unsqueeze(0).unsqueeze(2).expand(B, -1, L, -1)  # (C_A, D) to (B, C_A, L, D)

        x = torch.einsum("bcld->blcd", x)
        x = x.flatten(0, 1)  # (B * L, C_A, D)
        latents = torch.einsum("bcld->blcd", latents)
        latents = latents.flatten(0, 1)  # (B * L, C_A, D)

        x = self.level_agg(latents, x)  # (B * L, C, D)
        x = x.unflatten(dim=0, sizes=(B, L))  # (B, L, C, D)
        x = torch.einsum("blcd->bcld", x)  # (B, C, L, D)
        return x

    def forward(self, x: torch.Tensor):

        B, C, L, H, W = x.shape
        x = rearrange(x, "B C L H W -> B L C H W")

        embed_variable_list = []
        for i in range(self.levels):
            embed_variable_list.append(self.token_embeds[i](x[:,i]))

        embed_variable = torch.stack(embed_variable_list, dim=1)
        embed_variable = rearrange(embed_variable, "B L C D-> (B L) C D")

        x = embed_variable + self.pos_embed
        x = rearrange(x, "(B L) C D-> B L C D", B=B)

        x = self.aggregate_levels(x)  
        x = self.dropout(x)

        return x



if __name__=="__main__":
    encoder = Encoder((32,64))
    input = torch.rand(1, 19, 30, 32, 64)
    print(encoder(input).shape)