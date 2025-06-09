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
from poseidon.embedding import FourierExpansion
from poseidon.utils import get_2d_sincos_pos_embed, get_1d_sincos_pos_embed_from_grid, get_pad2d

__all__=["Decoder"]

class Decoder(nn.Module):
    """Multi-scale multi-source multi-variable decoder based on the Perceiver architecture."""

    def __init__(
        self,
        img_size,
        variables=19,
        levels=30,
        patch_size=2,
        embed_dim=1024,
        num_heads=16,
    ) -> None:
        """Initialise.

        Args:
        """
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.variables = variables
        self.levels = levels
        self.embed_dim = embed_dim

        self.level_decoder = PerceiverResampler(
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
        
        self.level_expansion = FourierExpansion(1, self.levels, d=embed_dim)
        self.head = nn.ModuleList([FinalLayer(embed_dim, patch_size, variables) for i in range(self.levels)])

        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        for i in range(self.levels):
            nn.init.constant_(self.head[i].adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.head[i].adaLN_modulation[-1].bias, 0)
            nn.init.constant_(self.head[i].linear.weight, 0)
            nn.init.constant_(self.head[i].linear.bias, 0)

    def deaggregate_levels(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Deaggregate pressure level information.

        Args:
            level_embed (torch.Tensor): Level embedding of shape `(B, L, C, D)`.
            x (torch.Tensor): Aggregated input of shape `(B, L, C', D)`.
            level_decoder (nn.Module): Pressure level decoder.

        Returns:
            torch.Tensor: Deaggregate output of shape `(B, L, C, D)`.
        """
        B, _, L, _ = x.shape

        levels = torch.arange(1, self.levels+1, device=x.device).float()
        latents = self.level_expansion(levels)
        latents = latents.unsqueeze(0).expand(B*L, -1, -1)  # (C_A, D) to (B, C_A, L, D)

        x = torch.einsum("bcld->blcd", x)
        x = x.flatten(0, 1)  # (BxL, C', D)

        x = self.level_decoder(latents, x)  # (BxL, C, D)
        x = x.unflatten(dim=0, sizes=(B, L))  # (B, L, C, D)
        x = torch.einsum("blcd->bcld", x)  # (B, C, L, D)
        return x
    
    def unpatchify(self, x: torch.Tensor, h=None, w=None):
        """
        x: (B, L, V * patch_size**2)
        return imgs: (B, V, H, W)
        """
        p = self.patch_size
        v = self.variables 
        h = self.img_size[0] // p if h is None else h // p
        w = self.img_size[1] // p if w is None else w // p
        assert h * w == x.shape[1],print(x.shape, h, w)

        x = x.reshape(shape=(x.shape[0], h, w, p, p, v))
        x = torch.einsum("nhwpqv->nvhpwq", x)
        imgs = x.reshape(shape=(x.shape[0], v, h * p, w * p))
        return imgs

    def forward(
        self,
        x: torch.Tensor,
        y_time_emb=None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Backbone output of shape `(B, L, D)`.

        Returns:
            :class:`aurora.batch.Batch`: Prediction for `batch`.
        """
        B, L, C, D = x.shape
        x = self.deaggregate_levels(x)  

        x_list = []
        for i in range(self.levels):
            x_list.append(self.head[i](x[:,i], y_time_emb))

        x = torch.stack(x_list, dim=1)
        x = rearrange(x, "B L C D-> (B L) C D")
        x = self.unpatchify(x)
        x = rearrange(x, "(B L) C H W-> B C L H W", L=self.levels)

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

    def forward(self, x, c=None):
        if c is not None:
            shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
            x = self.norm_final(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        x = self.linear(x)
        return x
    

if __name__=="__main__":
    decoder = Decoder((33,65))
    input = torch.rand(1,8,512,1024)
    y_mark = torch.Tensor([[11, 30]])
    print(decoder(input, y_mark).shape)