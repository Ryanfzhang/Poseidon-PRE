import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import trunc_normal_, Mlp, PatchEmbed
from functools import lru_cache
import numpy as np
import os
import sys
from einops import rearrange
sys.path.insert(0, os.path.join(os.getcwd(), "../"))

from poseidon.encoder import Encoder 
from poseidon.decoder import Decoder
from poseidon.embedding import FourierExpansion
from poseidon.backbone import Block


class poseidon_recon(nn.Module):
    def __init__(self, 
        in_img_size=(400,441),
        variables=19,
        n_level=30,
        patch_size=4,
        hidden_size=1024,
        depth=24,
        num_heads=8,
        mlp_ratio=4.0,
        latent_levels=4
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
        self.n_level = n_level
        self.patch_size = patch_size
        
        # embedding
        self.encoder = Encoder(img_size=in_img_size, patch_size=patch_size, embed_dim=hidden_size, latent_levels=latent_levels)

        self.month_expansion = FourierExpansion(1, 12, d=hidden_size)
        self.day_expansion = FourierExpansion(1, 31, d=hidden_size)
        
        # self.blocks = nn.ModuleList([
        #     Block(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        # ])
        
        self.decoder = Decoder(img_size=in_img_size, patch_size=patch_size, embed_dim=hidden_size)

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
        # Zero-out adaLN modulation layers in blocks:
        # for block in self.blocks:
            # nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            # nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.decoder.head.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.decoder.head.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.decoder.head.linear.weight, 0)
        nn.init.constant_(self.decoder.head.linear.bias, 0)

    def forward(self, x, x_mark, y_mark):
        B, C, N, H, W = x.shape

        if self.padding:
            x = self.pad(x)
        
        x = self.encoder(x) # B, L, D
        x = rearrange(x, "B L C D ->(B L) C D")

        x_time_emb = self.month_expansion(x_mark[:, 0]) + self.day_expansion(x_mark[:, 1])
        y_time_emb = self.month_expansion(y_mark[:, 0]) + self.day_expansion(y_mark[:, 1])

        # for block in self.blocks:
            # x = block(x, x_time_emb)
        
        x = rearrange(x, "(B L) C D ->B L C D", B=B)
        x = self.decoder(x)
        x = x[:,:,:,self.pad_size_h:, self.pad_size_w:]
        
        return x
    
class poseidon_pre(nn.Module):
    def __init__(self, 
        in_img_size=(400,441),
        variables=19,
        n_level=30,
        patch_size=4,
        hidden_size=1024,
        depth=24,
        num_heads=8,
        mlp_ratio=4.0,
        latent_levels=4
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
        self.n_level = n_level
        self.patch_size = patch_size
        
        # embedding
        self.encoder = Encoder(img_size=in_img_size, patch_size=patch_size, embed_dim=hidden_size, latent_levels=latent_levels)

        self.month_expansion = FourierExpansion(1, 12, d=hidden_size)
        self.day_expansion = FourierExpansion(1, 31, d=hidden_size)
        
        self.blocks = nn.ModuleList([
            Block(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        
        self.decoder = Decoder(img_size=in_img_size, patch_size=patch_size, embed_dim=hidden_size)

        self.initialize_weights()

        self.encoder.requires_grad_=False
        self.decoder.requires_grad_=False

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
        # Zero-out adaLN modulation layers in blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.decoder.head.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.decoder.head.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.decoder.head.linear.weight, 0)
        nn.init.constant_(self.decoder.head.linear.bias, 0)

    def forward(self, x, x_mark, y_mark):
        B, C, N, H, W = x.shape

        if self.padding:
            x = self.pad(x)
        
        x = self.encoder(x) # B, L, D
        x = rearrange(x, "B L C D ->(B L) C D")

        x_time_emb = self.month_expansion(x_mark[:, 0]) + self.day_expansion(x_mark[:, 1])
        y_time_emb = self.month_expansion(y_mark[:, 0]) + self.day_expansion(y_mark[:, 1])

        for block in self.blocks:
            x = block(x, x_time_emb)
        
        x = rearrange(x, "(B L) C D ->B L C D", B=B)
        x = self.decoder(x)
        x = x[:,:,:,self.pad_size_h:, self.pad_size_w:]
        
        return x

if __name__=="__main__":
    model = poseidon_recon(in_img_size=(400, 441), patch_size=8).to("cuda")
    x = torch.randn(1, 19, 30, 400, 441).to("cuda")
    x_mark = torch.Tensor([[11, 30]]).to("cuda")
    y_mark = torch.Tensor([[11, 30]]).to("cuda")
    y = model(x, x_mark, y_mark)
    print(y.shape)