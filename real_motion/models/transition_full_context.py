"""P0-F4 OccFM transition with larger full-history context conditioning."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from forecast.models.modules.transition_models.FlowMatchingV2 import FLOW_MATCHING_DOWN_X4_DiT
from forecast.models.modules.base.postion_embed import timestep_embedding, get_2d_sincos_pos_embed
from .blocks import SpatialAdaLNDiTBlock


class MotionWindowFlowMatchingFullContext(FLOW_MATCHING_DOWN_X4_DiT):
    """OccFM local transition + cheap surrounding-history context branch.

    The future state remains 20x20. A 40x40 full-history crop is temporally
    summarized and projected with one zero-initialized stride-2 3x3 conv. This
    adds scene context without running a second dense future world model.
    """

    def __init__(
        self,
        *args,
        prior_channels=32,
        context_channels=16,
        full_input_size=(50, 50),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        time_dim = self.model_channels
        self.prior_proj = nn.Linear(prior_channels, time_dim, bias=True)
        nn.init.zeros_(self.prior_proj.weight)
        nn.init.zeros_(self.prior_proj.bias)
        self.context_proj = nn.Conv2d(
            int(context_channels), time_dim, kernel_size=3, stride=2, padding=1, bias=True
        )
        nn.init.zeros_(self.context_proj.weight)
        nn.init.zeros_(self.context_proj.bias)

        self.full_input_size = tuple(full_input_size)
        if self.full_input_size[0] != self.full_input_size[1]:
            raise ValueError("absolute sin-cos embedding expects a square full latent grid")
        n_full = self.full_input_size[0] * self.full_input_size[1]
        official_grid = int(n_full ** 0.5) + 1
        abs_pos = get_2d_sincos_pos_embed(self.model_channels, official_grid)[:n_full]
        self.register_buffer(
            "absolute_pos_embed",
            torch.from_numpy(abs_pos).float().reshape(
                1, self.full_input_size[0], self.full_input_size[1], self.model_channels
            ),
            persistent=False,
        )
        self._upgrade_dit_blocks()

    def _convert_block(self, old):
        new = SpatialAdaLNDiTBlock(
            hidden_size=old.norm1.normalized_shape[0],
            num_heads=old.attn.num_heads if hasattr(old.attn, "num_heads") else old.attn.num_heads,
            cond_size=old.adaLN_modulation[-1].in_features,
            attention_mode=getattr(old.attn, "attention_mode", "flash"),
        )
        new.load_state_dict(old.state_dict(), strict=True)
        return new

    def _upgrade_dit_blocks(self):
        for group in self.downs:
            group[2] = self._convert_block(group[2])
            group[3] = self._convert_block(group[3])
        for group in self.ups:
            group[2] = self._convert_block(group[2])
            group[3] = self._convert_block(group[3])

    @staticmethod
    def _resize_prior(prior_bfchw, hw):
        b, f, c, h, w = prior_bfchw.shape
        y = F.interpolate(
            prior_bfchw.reshape(b * f, c, h, w),
            size=hw,
            mode="bilinear",
            align_corners=False,
        )
        return y.reshape(b, f, c, *hw)

    def _context_base(self, history_context, batch_size, device, dtype):
        if history_context is None:
            return None
        if history_context.ndim != 5 or history_context.shape[0] != batch_size:
            raise ValueError("history_context must be [B,T,C,H,W]")
        if history_context.shape[2] != self.context_proj.in_channels:
            raise ValueError("history_context channel mismatch")
        ctx = history_context.to(device=device, dtype=dtype).mean(dim=1)
        return self.context_proj(ctx)

    def _context_spatial(self, context_base, hw, frames):
        if context_base is None:
            return None
        c = F.interpolate(context_base, size=hw, mode="bilinear", align_corners=False)
        c = rearrange(c, "b d h w -> b (h w) d")
        return repeat(c, "b n d -> (b f) n d", f=frames)

    def _context_temporal(self, context_base, hw, frames):
        if context_base is None:
            return None
        c = F.interpolate(context_base, size=hw, mode="bilinear", align_corners=False)
        c = rearrange(c, "b d h w -> (b h w) d")
        return repeat(c, "bn d -> bn f d", f=frames)

    def _mid_physics_fusion(self, x, prior_future):
        """Extension hook used by P0-F9; legacy transitions are exact no-ops."""
        return x

    def _spatial_cond(self, emb, prior, hw, context_base=None):
        p = self._resize_prior(prior, hw)
        p = rearrange(p, "b f c h w -> (b f) (h w) c")
        cond = self.prior_proj(p) + repeat(
            emb, "b d -> (b f) n d", f=prior.shape[1], n=hw[0] * hw[1]
        )
        ctx = self._context_spatial(context_base, hw, prior.shape[1])
        return cond if ctx is None else cond + ctx

    def _temporal_cond(self, emb, prior, hw, context_base=None):
        p = self._resize_prior(prior, hw)
        p = rearrange(p, "b f c h w -> (b h w) f c")
        cond = self.prior_proj(p) + repeat(
            emb, "b d -> (b n) f d", n=hw[0] * hw[1], f=prior.shape[1]
        )
        ctx = self._context_temporal(context_base, hw, prior.shape[1])
        return cond if ctx is None else cond + ctx

    def _window_pos(self, origins, batch_size, frames, hw, device, dtype):
        if origins is None:
            if self.pos_embed.shape[1] != hw[0] * hw[1]:
                raise RuntimeError("local pos_embed shape mismatch")
            return self.pos_embed.to(device=device, dtype=dtype)
        if origins.shape != (batch_size, 2):
            raise ValueError(f"window_origins must be [B,2], got {tuple(origins.shape)}")
        ph, pw = hw
        origins = origins.to(device=self.absolute_pos_embed.device, dtype=torch.long)
        max_y = self.absolute_pos_embed.shape[1] - ph
        max_x = self.absolute_pos_embed.shape[2] - pw
        if bool(
            (
                (origins[:, 0] < 0)
                | (origins[:, 0] > max_y)
                | (origins[:, 1] < 0)
                | (origins[:, 1] > max_x)
            ).any()
        ):
            raise ValueError("window origin out of full latent bounds")
        yy = origins[:, 0, None, None] + torch.arange(ph, device=origins.device)[None, :, None]
        xx = origins[:, 1, None, None] + torch.arange(pw, device=origins.device)[None, None, :]
        grid = self.absolute_pos_embed[0]
        pos = grid[yy.expand(-1, -1, pw), xx.expand(-1, ph, -1)]
        pos = pos.reshape(batch_size, ph * pw, -1).to(device=device, dtype=dtype)
        return repeat(pos, "b n d -> (b f) n d", f=frames)

    def forward_single(
        self,
        x,
        timesteps=None,
        trajectory=None,
        prior_condition=None,
        history_context=None,
        window_origins=None,
        **kwargs,
    ):
        if prior_condition is None:
            prior_condition = torch.zeros(
                x.shape[0],
                x.shape[1],
                self.prior_proj.in_features,
                x.shape[-2],
                x.shape[-1],
                dtype=x.dtype,
                device=x.device,
            )
        if prior_condition.shape[:2] != x.shape[:2] or prior_condition.shape[-2:] != x.shape[-2:]:
            raise ValueError("prior_condition must align with [B,F,...,H,W] of x")

        batch_size = x.shape[0]
        context_base = self._context_base(history_context, batch_size, x.device, x.dtype)
        x = rearrange(x, "b f c h w -> b c f h w")
        if x.shape[2] > self.temp_embed.shape[1]:
            raise ValueError(
                f"sequence length {x.shape[2]} exceeds inherited temp_embed length {self.temp_embed.shape[1]}"
            )
        time_rel_pos_bias = self.time_rel_pos_bias(x.shape[2], device=x.device)
        x = self.init_conv(x)
        x = x + self.init_temporal_attn(x, pos_bias=time_rel_pos_bias)
        r = x.clone()

        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.t_embedder(t_emb)
        if trajectory is not None:
            trajectory = trajectory[:, : self.traj_length, :]
            emb = emb + self.encode_pose(trajectory)

        hskip = []
        for idx, (block1, block2, spatial_attn, temporal_attn, identity, downsample) in enumerate(self.downs):
            x = block1(x, emb)
            x = block2(x, emb)
            hh, ww = x.shape[-2:]
            xs = rearrange(x, "b c f h w -> (b f) (h w) c")
            if idx == 0:
                xs = xs + self._window_pos(
                    window_origins, r.shape[0], x.shape[2], (hh, ww), xs.device, xs.dtype
                )
            xs = spatial_attn(xs, self._spatial_cond(emb, prior_condition, (hh, ww), context_base))
            xt = rearrange(xs, "(b f) n c -> (b n) f c", b=r.shape[0], f=x.shape[2])
            if idx == 0:
                te = self.temp_embed[:, : xt.shape[1], :]
                xt = xt + te.repeat(r.shape[0] * hh * ww, 1, 1)
            xt = temporal_attn(xt, self._temporal_cond(emb, prior_condition, (hh, ww), context_base))
            x = rearrange(xt, "(b h w) f c -> b c f h w", b=r.shape[0], h=hh, w=ww)
            hskip.append(x)
            x = downsample(x)

        x = self.mid_block1(x, emb)
        x = self._mid_physics_fusion(x, prior_condition)
        x = self.mid_spatial_attn(x)
        x = self.mid_temporal_attn(x, pos_bias=time_rel_pos_bias)
        x = self.mid_block2(x, emb)

        for block1, block2, spatial_attn, temporal_attn, upsample in self.ups:
            x = torch.cat((x, hskip.pop()), dim=1)
            x = block1(x, emb)
            x = block2(x, emb)
            hh, ww = x.shape[-2:]
            xs = rearrange(x, "b c f h w -> (b f) (h w) c")
            xs = spatial_attn(xs, self._spatial_cond(emb, prior_condition, (hh, ww), context_base))
            xt = rearrange(xs, "(b f) n c -> (b n) f c", b=r.shape[0], f=x.shape[2])
            xt = temporal_attn(xt, self._temporal_cond(emb, prior_condition, (hh, ww), context_base))
            x = rearrange(xt, "(b h w) f c -> b c f h w", b=r.shape[0], h=hh, w=ww)
            x = upsample(x)

        x = self.final_conv(torch.cat((x, r), dim=1))
        return rearrange(x, "b c f h w -> b f c h w")

    def forward(self, batch_dict):
        batch_dict["predicted_latent"] = self.forward_single(
            batch_dict["noised_sequence"],
            batch_dict["timesteps"],
            batch_dict.get("trajectory"),
            prior_condition=batch_dict.get("prior_condition"),
            history_context=batch_dict.get("history_context"),
            window_origins=batch_dict.get("window_origins"),
        )
        return batch_dict
