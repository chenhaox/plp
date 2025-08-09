"""
Created on 2025/8/8 
Author: Hao Chen (chen960216@gmail.com)
"""
import math
from dataclasses import dataclass
from typing import Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetBlock(nn.Module):
    def __init__(self, num_filters: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, 1, 1, bias=False)
        self.act = nn.ELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        out = self.act(out + x)
        return out


class MLP(nn.Module):
    def __init__(self, dims: List[int], act=nn.ELU):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1])]
            if i < len(dims) - 2:
                layers += [act()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ManifestEncoder(nn.Module):
    """
    Input: items as tensor [B, N, 4] with columns (w, d, h, c)
    Output: pooled embedding g [B, E]
    """

    def __init__(self, phi_hidden=128, emb_dim=128):
        super().__init__()
        in_feats = 8  # w_min, d_max, h, area, vol, aspect, is_square, count
        self.phi = MLP([in_feats, phi_hidden, phi_hidden])
        # Pool: weighted sum and max, then rho
        self.rho = MLP([2 * phi_hidden, emb_dim])

    def forward(self, items: torch.Tensor) -> torch.Tensor:
        # items: [B, N, 4] -> (w, d, h, c)
        w, d, h, c = items.unbind(dim=-1)
        w_min = torch.minimum(w, d)
        d_max = torch.maximum(w, d)
        area = w * d
        vol = area * h
        aspect = w_min / (d_max.clamp_min(1e-6))
        is_square = (w == d).float()

        feats = torch.stack([w_min, d_max, h, area, vol, aspect, is_square, c], dim=-1)  # [B,N,8]
        phi_x = self.phi(feats)  # [B,N,H]

        # weighted sum by counts and max
        sum_pool = (phi_x * c.unsqueeze(-1)).sum(dim=1)  # [B, H]
        max_pool, _ = phi_x.max(dim=1)  # [B, H]
        pooled = torch.cat([sum_pool, max_pool], dim=-1)  # [B, 2H]
        g = self.rho(pooled)  # [B, E]
        return g


# ---------- FiLM conditioning ----------

class FiLM(nn.Module):
    def __init__(self, feat_channels: int, emb_dim: int):
        super().__init__()
        self.to_gamma = nn.Linear(emb_dim, feat_channels)
        self.to_beta = nn.Linear(emb_dim, feat_channels)

    def forward(self, feat: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        # feat: [B, C, H, W], g: [B, E]
        gamma = self.to_gamma(g).unsqueeze(-1).unsqueeze(-1)  # [B,C,1,1]
        beta = self.to_beta(g).unsqueeze(-1).unsqueeze(-1)  # [B,C,1,1]
        return feat * (1 + gamma) + beta


class D3QN(nn.Module):
    """
    Dueling head; acts like a standard Q-network at inference:
    forward(obs) -> Q(s, ·)   (shape [B, num_actions])
    """

    def __init__(
            self,
            grid_dim: Tuple[int, int, int],  # (C, H, W)
            num_actions: int,
            num_res_block: int = 8,
            num_filters: int = 128,
            head_channels_adv: int = 8,
            head_channels_val: int = 8,
            manifest_emb_dim: int = 128,
    ) -> None:
        super().__init__()
        self.W, self.D, self.H_max = grid_dim
        self.num_actions = num_actions
        C_in = 6
        self.stem = nn.Sequential(
            nn.Conv2d(C_in, num_filters, 3, 1, 1, bias=False),
            nn.ELU(),
        )
        self.body = nn.Sequential(*[ResNetBlock(num_filters) for _ in range(num_res_block)])

        # DeepSets encoder + FiLM
        self.manifest_enc = ManifestEncoder(phi_hidden=128, emb_dim=manifest_emb_dim)
        self.film = FiLM(num_filters, manifest_emb_dim)

        flat = self.W * self.D
        self.adv_head = nn.Sequential(
            nn.Conv2d(num_filters, head_channels_adv, 1, 1, bias=False),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(head_channels_adv * flat, self.num_actions),
        )
        self.val_head = nn.Sequential(
            nn.Conv2d(num_filters, head_channels_val, 1, 1, bias=False),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(head_channels_val * flat, 1),
        )

        # Register coord maps (buffers so they travel with device/dtype)
        xx, yy = torch.meshgrid(
            torch.linspace(0, 1, self.W),
            torch.linspace(0, 1, self.D),
            indexing='ij'
        )
        self.register_buffer("coord_x", xx.clone().unsqueeze(0).unsqueeze(0))  # [1,1,H,W]
        self.register_buffer("coord_y", yy.clone().unsqueeze(0).unsqueeze(0))

    def _make_spatial_input(
            self,
            height_map: torch.Tensor,  # [B,1,H,W], raw integer heights
            pd_xy: torch.Tensor,  # [B,2] (x_start, y_start)  0-based
            z_start: torch.Tensor,  # [B,] or [B,1]
    ) -> torch.Tensor:
        B = height_map.shape[0]
        W, D = self.W, self.D

        height_norm = height_map / float(self.H_max)
        free_norm = (self.H_max - height_map) / float(self.H_max)

        # pd mask
        pd_mask = torch.zeros((B, 1, W, D), device=height_map.device, dtype=height_map.dtype)
        xs = pd_xy[:, 0].long().clamp(0, W - 1)
        ys = pd_xy[:, 1].long().clamp(0, D - 1)
        pd_mask[torch.arange(B), 0, xs, ys] = 1.0

        # z-plane
        z_norm = (z_start.view(B, 1, 1, 1).to(height_map.dtype)) / float(self.H_max)
        z_plane = z_norm.expand(B, 1, W, D)

        # coord channels
        coord_x = self.coord_x.expand(B, -1, -1, -1)
        coord_y = self.coord_y.expand(B, -1, -1, -1)

        x = torch.cat([height_norm, free_norm, pd_mask, z_plane, coord_x, coord_y], dim=1)
        return x

    def forward(self,
                height_map: torch.Tensor,  # [B,1,20,20]
                dp_xy: torch.Tensor,  # [B,2]
                z_start: torch.Tensor,  # [B]
                manifest: torch.Tensor,  # [B,N,4] (w,d,h,c)
                ) -> torch.Tensor:
        g = self.manifest_enc(manifest)  # [B,E]
        x0 = self._make_spatial_input(height_map, dp_xy, z_start)
        f = self.body(self.stem(x0))
        f = self.film(f, g)  # condition on manifest
        advantage = self.adv_head(f)  # [B, A]
        value = self.val_head(f)  # [B, 1]
        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q
