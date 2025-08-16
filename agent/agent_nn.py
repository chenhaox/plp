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

    def __init__(self, W, D, H_max, phi_hidden=128, emb_dim=128):
        super().__init__()
        in_feats = 8  # w_min, d_max, h, area, vol, aspect, is_square, count
        self.W = W
        self.D = D
        self.H_max = H_max
        self.phi = MLP([in_feats, phi_hidden, phi_hidden])
        # Pool: weighted sum and max, then rho
        self.rho = MLP([2 * phi_hidden, emb_dim])

    def forward(self, items: torch.Tensor) -> torch.Tensor:
        # items: [B, N, 4] -> (w, d, h, c)
        w, d, h, c = items.unbind(dim=-1)
        w_norm = w / self.W  # normalize by W
        d_norm = d / self.D  # normalize by D
        h_norm = h / self.H_max  # normalize by H_max
        w_min = torch.minimum(w_norm, d_norm)
        d_max = torch.maximum(w_norm, d_norm)
        area = w_min * d_max  # area = min(w, d) * max(w, d)
        vol = area * h_norm  # volume = area * h
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
        C_in = 5
        self.stem = nn.Sequential(
            nn.Conv2d(C_in, num_filters, 3, 1, 1, bias=False),
            nn.ELU(),
        )
        self.body = nn.Sequential(*[ResNetBlock(num_filters) for _ in range(num_res_block)])

        # DeepSets encoder + FiLM
        # self.manifest_enc = ManifestEncoder(
        #     W=self.W,
        #     D=self.D,
        #     H_max=self.H_max,
        #     phi_hidden=manifest_emb_dim,
        #     emb_dim=manifest_emb_dim)
        # self.film = FiLM(num_filters, manifest_emb_dim)

        flat = self.W * self.D
        # self.adv_head = nn.Sequential(
        #     nn.Conv2d(num_filters, head_channels_adv, 1, 1, bias=False),
        #     nn.ELU(),
        #     nn.Flatten(),
        #     nn.Linear(head_channels_adv * flat, self.num_actions),
        # )
        # self.val_head = nn.Sequential(
        #     nn.Conv2d(num_filters, head_channels_val, 1, 1, bias=False),
        #     nn.ELU(),
        #     nn.Flatten(),
        #     nn.Linear(head_channels_val * flat, 1),
        # )

        # Advantage head (spatial): [B, C, W, D] -> [B, 1, W, D] -> [B, W*D]
        self.adv_head_spatial = nn.Sequential(
            nn.Conv2d(num_filters, head_channels_adv, kernel_size=1, bias=False),
            nn.ELU(),
            nn.Conv2d(head_channels_adv, 1, kernel_size=1, bias=True),
        )

        # Value head (global): [B, C, W, D] -> GAP -> [B, C] -> [B, 1]
        self.val_head_global = nn.Sequential(
            nn.Conv2d(num_filters, head_channels_val, kernel_size=1, bias=False),
            nn.ELU(),
        )
        self.val_fc = nn.Linear(head_channels_val, 1)
        # DoNothing advantage logit from pooled features
        self.pool_for_dn = nn.AdaptiveAvgPool2d((1, 1))
        self.donothing_fc = nn.Linear(num_filters, 1)

        # Register coord maps (buffers so they travel with device/dtype)
        xx, yy = torch.meshgrid(
            torch.linspace(0, 1, self.W),
            torch.linspace(0, 1, self.D),
            indexing='ij'
        )
        # self.register_buffer("coord_x", xx.clone().unsqueeze(0).unsqueeze(0))  # [1,1,H,W]
        # self.register_buffer("coord_y", yy.clone().unsqueeze(0).unsqueeze(0))

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
        # pd_mask = torch.zeros((B, 1, W, D), device=height_map.device, dtype=height_map.dtype)
        # xs = pd_xy[:, 0].long().clamp(0, W - 1)
        # ys = pd_xy[:, 1].long().clamp(0, D - 1)
        # pd_mask[torch.arange(B), 0, xs, ys] = 1.0

        # normalized decision-point planes (slight fix: divide by (W-1),(D-1))
        xs = (pd_xy[:, 0].float() / max(W - 1, 1)).view(B, 1, 1, 1)
        ys = (pd_xy[:, 1].float() / max(D - 1, 1)).view(B, 1, 1, 1)
        xs_plane = xs.expand(B, 1, W, D)
        ys_plane = ys.expand(B, 1, W, D)

        z_plane = (z_start.view(B, 1, 1, 1).float() / float(self.H_max)).expand(B, 1, W, D)

        # coord channels
        # coord_x = self.coord_x.expand(B, -1, -1, -1)
        # coord_y = self.coord_y.expand(B, -1, -1, -1)

        # x = torch.cat([height_norm, free_norm, pd_mask, z_plane, coord_x, coord_y], dim=1)
        x = torch.cat([height_norm, free_norm, xs_plane, ys_plane, z_plane, ], dim=1)
        return x

    def forward(self,
                height_map: torch.Tensor,  # [B,1,20,20]
                dp_xy: torch.Tensor,  # [B,2]
                z_start: torch.Tensor,  # [B]
                manifest: torch.Tensor,  # [B,N,4] (w,d,h,c)
                action_mask: Optional[torch.Tensor] = None,  # [B, W*D+1] boolean; True=valid
                ) -> torch.Tensor:
        # g = self.manifest_enc(manifest)  # [B,E]
        x0 = self._make_spatial_input(height_map, dp_xy, z_start)
        f = self.body(self.stem(x0))
        # f = self.film(f, g)  # condition on manifest
        # advantage = self.adv_head(f)  # [B, A]
        # value = self.val_head(f)  # [B, 1]
        # q = value + advantage - advantage.mean(dim=1, keepdim=True)
        # Per-cell advantages (flatten to [B, W*D])
        adv_map = self.adv_head_spatial(f).squeeze(1)  # [B, W, D]
        adv_flat = adv_map.flatten(start_dim=1)  # [B, W*D]

        # DoNothing advantage
        pooled = self.pool_for_dn(f).squeeze(-1).squeeze(-1)  # [B, C]
        adv_dn = self.donothing_fc(pooled)  # [B, 1]

        advantage = torch.cat([adv_flat, adv_dn], dim=1)  # [B, W*D+1]

        # State value
        v_feat = self.val_head_global(f)  # [B, C', W, D]
        v_gap = v_feat.mean(dim=(2, 3))  # [B, C']
        value = self.val_fc(v_gap)  # [B, 1]
        # Dueling combine with MASK-AWARE mean subtraction
        if action_mask is not None:
            # prevent divide-by-zero if mask is all False (rare; then default to plain mean)
            valid_counts = action_mask.sum(dim=1, keepdim=True).clamp_min(1)
            masked_adv_mean = (advantage * action_mask).sum(dim=1, keepdim=True) / valid_counts
            q = value + advantage - masked_adv_mean
            # hard-mask invalid actions to a big negative number
            very_neg = torch.finfo(q.dtype).min / 8  # stable large negative
            q = torch.where(action_mask, q, very_neg)
        else:
            q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q
