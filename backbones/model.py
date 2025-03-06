import sys
import os
sys.path.insert(0, os.getcwd())
from backbones.patch import PatchEmbed2D, PatchEmbed3D, PatchRecovery2D, PatchRecovery3D
from backbones.layer import *
import torch

class OceanTransformer(nn.Module):
    def __init__(self, 
                 lon_resolution=441,
                 lat_resolution=400,
                 emb_dim=32, 
                 cond_dim=32, # dim of the conditioning
                 two_poles=False,
                 num_heads=(1, 2, 2, 1), 
                 droppath_coeff=0.2,
                 patch_size=(4, 8, 8),
                 window_size=(2, 6, 12), 
                 depth_multiplier=1,
                 position_embs_dim=4,
                 surface_ch=0,
                 level_ch=30,
                 n_level_variables=19,
                 use_prev=False, 
                 use_skip=False,
                 conv_head=False,
                 freeze_backbone=False,
                 dropout=0.0,
                 first_interaction_layer=False,
                 gradient_checkpointing=False,
                **kwargs):
        super().__init__()
        self.__dict__.update(locals())
        drop_path = np.linspace(0, droppath_coeff/depth_multiplier, 8*depth_multiplier).tolist()

        # In addition, three constant masks(the topography mask, land-sea mask and soil type mask)

        self.layer1_shape = (8, 50, 56)
        
        self.layer2_shape = (8, 25, 28)
        
        self.positional_embeddings = nn.Parameter(torch.zeros((position_embs_dim, lat_resolution, lon_resolution)))
        torch.nn.init.trunc_normal_(self.positional_embeddings, 0.02)
        
        
        # self.patchembed2d = PatchEmbed2D(
        #     img_size=(lat_resolution, lon_resolution),
        #     patch_size=patch_size[1:],
        #     in_chans=surface_ch + position_embs_dim,  # add
        #     embed_dim=emb_dim,
        # )
        self.patchembed3d = PatchEmbed3D(
            img_size=(30, lat_resolution, lon_resolution),
            patch_size=patch_size,
            in_chans=n_level_variables+position_embs_dim,
            embed_dim=emb_dim
        )

        if first_interaction_layer == 'linear':
            self.interaction_layer = LinVert(in_features=emb_dim)

        act_layer1 = act_layer2 = act_layer4 = nn.GELU

        self.layer1 = BasicLayer(
            dim=emb_dim,
            input_resolution=self.layer1_shape,
            depth=2*depth_multiplier,
            num_heads=num_heads[0],
            window_size=window_size,
            drop_path=drop_path[:2*depth_multiplier],
            act_layer=act_layer1,
            drop=dropout,
            **kwargs
        )
        self.downsample = DownSample(in_dim=emb_dim, input_resolution=self.layer1_shape, output_resolution=self.layer2_shape)
        self.layer2 = BasicLayer(
            dim=emb_dim * 2,
            input_resolution=self.layer2_shape, 
            depth=6*depth_multiplier,
            num_heads=num_heads[1],
            window_size=window_size,
            drop_path=drop_path[2*depth_multiplier:],
            act_layer=act_layer2,
            drop=dropout,
            **kwargs
        )
        self.layer3 = BasicLayer(
            dim=emb_dim * 2,
            input_resolution=self.layer2_shape,
            depth=6*depth_multiplier,
            num_heads=num_heads[2],
            window_size=window_size,
            drop_path=drop_path[2*depth_multiplier:],
            act_layer=act_layer2,
            drop=dropout,
            **kwargs
        )
        self.upsample = UpSample(emb_dim * 2, emb_dim, self.layer2_shape, self.layer1_shape)
        out_dim = emb_dim if not self.use_skip else 2*emb_dim
        self.layer4 = BasicLayer(
            dim=out_dim,
            input_resolution=self.layer1_shape,
            depth=2*depth_multiplier,
            num_heads=num_heads[3],
            window_size=window_size,
            drop_path=drop_path[:2*depth_multiplier],
            act_layer=act_layer4,
            drop=dropout,
            **kwargs
        )
        # The outputs of the 2nd encoder layer and the 7th decoder layer are concatenated along the channel dimension.
        if self.freeze_backbone:
            for p in self.parameters():
                p.requires_grad = False
                
        if not self.conv_head:
            self.patchrecovery3d = PatchRecovery3D((30, lat_resolution, lon_resolution), patch_size, out_dim, n_level_variables)

            for p in self.patchembed3d.parameters():
                p.requires_grad = True

        if conv_head:
            self.patchrecovery = PatchRecovery5(input_dim=self.zdim*out_dim, output_dim=69, downfactor=patch_size[-1],
                                                n_level_variables=n_level_variables)
            for p in self.patchrecovery.parameters():
                p.requires_grad = True

    def forward(self, input_level, cond_emb=None, **kwargs):
        """
        Args:
            surface (torch.Tensor): 2D n_lat=721, n_lon=1440, chans=4.
            surface_mask (torch.Tensor): 2D n_lat=721, n_lon=1440, chans=3.
            upper_air (torch.Tensor): 3D n_pl=30, n_lat=400, n_lon=441, chans=19.
        """
        
        lat_res, lon_res = self.lat_resolution, self.lon_resolution
        init_shape = (input_level.shape[-2], input_level.shape[-1])

        pos_embs = self.positional_embeddings.unsqueeze(0).unsqueeze(2).expand((input_level.shape[0], -1, input_level.shape[2], -1, -1))
        x = torch.concat([input_level, pos_embs], dim=1)
        x = self.patchembed3d(x)

        B, C, Pl, Lat, Lon = x.shape
        x = x.reshape(B, C, -1).transpose(1, 2)

        if self.first_interaction_layer:
            x = self.interaction_layer(x)

        x = self.layer1(x, cond_emb)

        skip = x
        x = self.downsample(x)
        
        x = self.layer2(x, cond_emb)

        if self.gradient_checkpointing:
            x = gradient_checkpoint.checkpoint(self.layer3, x, cond_emb, use_reentrant=False)
        else:
            x = self.layer3(x, cond_emb)

        x = self.upsample(x)
        if self.use_skip and skip is not None:
            x = torch.concat([x, skip], dim=-1)
        x = self.layer4(x, cond_emb)

        output = x
        output = output.transpose(1, 2).reshape(output.shape[0], -1, *self.layer1_shape)
        
        if self.freeze_backbone:
            output = output.detach()
            
        output_level = self.patchrecovery3d(output)
            
        return output_level


if __name__=="__main__":
    model = OceanTransformer()
    input = torch.rand(1, 19, 30, 400, 441)
    print(model(input).shape)