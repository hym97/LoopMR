# SDUM's Restormer backbone (NVIDIA, "SDUM: A Scalable Deep Unrolled Model for Universal MRI Reconstruction", arXiv 2512.17137), block classes
# copied VERBATIM from github.com/NVIDIA-Medtech/NV-Raw2insights-MRI scripts/models/restormer/restormer.py (commit bc67bbc, Apache-2.0, see LICENSE):
# timestep_embedding, TimestepBlock, TimestepEmbedSequential, MDTA, GDFN, MoE, TransformerBlock, DownSample, UpSample, Restormer.
# Not copied: the cascade / CSME / data-consistency code, HybridRestormer, OCAB / relative-position attention, LabelEmbedder.
# The only non-verbatim lines are these imports: the original module imports monai (not installed here) for divisible_pad_t /
# inverse_divisible_pad_t only; the two stand-ins below zero-pad the trailing edge to a multiple of k and crop it back (our NormWrapper
# already pads to the block's pad_multiple, so they are no-ops in the loop).
import math
from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from timm.layers import DropPath


def divisible_pad_t(x, k=16):
    h, w = x.shape[-2:]
    ph, pw = (-h) % k, (-w) % k
    return (F.pad(x, (0, pw, 0, ph)) if (ph or pw) else x), (h, w)


def inverse_divisible_pad_t(x, padding_sizes):
    h, w = padding_sizes
    return x[..., :h, :w]


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=timesteps.device
        )
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, "b -> b d", d=dim)
    return embedding


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class MDTA(nn.Module):
    def __init__(self, channels, num_heads):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_conv = nn.Conv2d(
            channels * 3,
            channels * 3,
            kernel_size=3,
            padding=1,
            groups=channels * 3,
            bias=False,
        )
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv_conv(self.qkv(x)).chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)

        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)
        out = self.project_out(torch.matmul(attn, v).reshape(b, -1, h, w))
        return out


class GDFN(nn.Module):
    def __init__(self, channels, expansion_factor, emb_channels=None):
        super(GDFN, self).__init__()

        hidden_channels = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)
        self.conv = nn.Conv2d(
            hidden_channels * 2,
            hidden_channels * 2,
            kernel_size=3,
            padding=1,
            groups=hidden_channels * 2,
            bias=False,
        )
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)

        if emb_channels is not None:
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                nn.Linear(
                    emb_channels,
                    hidden_channels,
                ),
            )
        else:
            self.emb_layers = None

    def forward(self, x, emb=None):
        x1, x2 = self.conv(self.project_in(x)).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        if emb is not None and self.emb_layers is not None:
            emb_out = self.emb_layers(emb)
            while len(emb_out.shape) < len(x.shape):
                emb_out = emb_out[..., None]
            x = x + emb_out

        x = self.project_out(x)
        return x


class MoE(nn.Module):
    """
    A sparse Mixture of Experts (MoE) layer with top-k gating.

    This layer routes each input to a specific number of experts (`top_k`)
    and computes a weighted sum of their outputs.
    """

    def __init__(self, channels, expansion_factor, num_experts=4, top_k=1):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # Expert networks
        self.experts = nn.ModuleList([GDFN(channels, expansion_factor) for _ in range(self.num_experts)])

        # Gating network
        self.gate = nn.Linear(channels, self.num_experts)

    def forward(self, x):
        b, c, h, w = x.shape

        # 1. Use global average pooling to get a feature vector for routing
        gating_input = F.adaptive_avg_pool2d(x, (1, 1)).view(b, -1)

        # 2. Get router logits from the gate
        router_logits = self.gate(gating_input)  # Shape: (b, num_experts)

        # 3. Select top-k experts
        # Find the top_k scores and their indices
        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)  # Shape: (b, top_k)

        # 4. Re-normalize the weights of the top-k experts
        top_k_weights = F.softmax(top_k_logits, dim=-1)

        # 5. Create a sparse tensor of the final weights
        # Initialize a tensor of zeros with the same shape as the router logits
        sparse_weights = torch.zeros_like(router_logits)
        # Place the calculated top-k weights at the correct indices
        sparse_weights.scatter_(-1, top_k_indices, top_k_weights)  # In-place scatter

        # 6. Compute and combine expert outputs
        # Note: This implementation computes all experts but only uses the top-k ones.
        # For true computational savings, a more complex dispatching mechanism is needed.
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)

        # Reshape weights for broadcasting: (b, num_experts) -> (b, num_experts, 1, 1, 1)
        routing_weights = sparse_weights.view(b, self.num_experts, 1, 1, 1)

        # Compute the weighted sum of expert outputs (only top-k will have non-zero weights)
        weighted_output = torch.sum(routing_weights * expert_outputs, dim=1)

        return weighted_output


class TransformerBlock(TimestepBlock):
    def __init__(
        self,
        channels,
        num_heads,
        expansion_factor,
        num_experts=None,
        top_k=1,
        drop_path=0.0,
        norm_type=None,
        post_norm=False,
        emb_channels=None,
    ):
        super(TransformerBlock, self).__init__()

        self.norm_type = norm_type
        if self.norm_type == "ln":
            self.norm1 = nn.LayerNorm(channels)
            self.norm2 = nn.LayerNorm(channels)
        elif self.norm_type == "bn":
            self.norm1 = nn.BatchNorm2d(channels)
            self.norm2 = nn.BatchNorm2d(channels)
        elif self.norm_type is None:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
        else:
            raise NotImplementedError("Norm {} is not implemented. (bn or ln or null)".format(norm_type))
        self.attn = MDTA(channels, num_heads)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # Conditionally use MoE or a single GDFN
        if num_experts and num_experts > 0:
            self.ffn = MoE(channels, expansion_factor, num_experts, top_k)
        else:
            self.ffn = GDFN(channels, expansion_factor, emb_channels=emb_channels)

        self.post_norm = post_norm

    def forward(self, x, emb=None):
        b, c, h, w = x.shape
        if self.norm_type == "ln":
            if self.post_norm:
                x = x + self.norm1(
                    self.drop_path(self.attn(x)).reshape(b, c, -1).transpose(-2, -1).contiguous()
                ).transpose(-2, -1).contiguous().reshape(b, c, h, w)
                x = x + self.norm2(
                    self.drop_path(self.ffn(x, emb)).reshape(b, c, -1).transpose(-2, -1).contiguous()
                ).transpose(-2, -1).contiguous().reshape(b, c, h, w)
            else:
                x = x + self.drop_path(
                    self.attn(
                        self.norm1(x.reshape(b, c, -1).transpose(-2, -1).contiguous())
                        .transpose(-2, -1)
                        .contiguous()
                        .reshape(b, c, h, w)
                    )
                )
                x = x + self.drop_path(
                    self.ffn(
                        self.norm2(x.reshape(b, c, -1).transpose(-2, -1).contiguous())
                        .transpose(-2, -1)
                        .contiguous()
                        .reshape(b, c, h, w),
                        emb,
                    )
                )
        else:
            if self.post_norm:
                x = x + self.norm1(self.drop_path(self.attn(x)))
                x = x + self.norm2(self.drop_path(self.ffn(x, emb)))
            else:
                x = x + self.drop_path(self.attn(self.norm1(x)))
                x = x + self.drop_path(self.ffn(self.norm2(x), emb))
        return x


class DownSample(nn.Module):
    def __init__(self, channels):
        super(DownSample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class UpSample(nn.Module):
    def __init__(self, channels):
        super(UpSample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class Restormer(nn.Module):
    """
    A Restormer-style U-Net backbone that supports an arbitrary number of levels.

    Args:
        in_channel (int): input channels.
        out_channel (int): output channels.
        num_blocks (List[int]): number of TransformerBlocks per level (length L).
        num_heads  (List[int]): attention heads per level (length L).
        channels   (List[int]): channel width per level from top to bottom (length L, L >= 1).
        num_refinement (int): number of TransformerBlocks in final refinement at top level.
        expansion_factor, num_experts, top_k, drop_path, norm_type, post_norm: passed to TransformerBlock.

    cas_skips contract (same as original):
        - In forward(cas_skips=...), cas_skips is a list/tuple of length L-1.
        - cas_skips[i-1] is added to the encoder *input* at level i (i = 1..L-1).

    Returns:
        out: final image-like tensor.
        cas_outs (tuple of length L-1):
            [dec_out(level=1), ..., dec_out(level=L-2), enc_out(level=L-1)]
        For L=4, this matches original behavior: (out_dec2, out_dec3, out_enc4).
        For L=1, this is (out_dec1).
    """

    def __init__(
        self,
        in_channel=2,
        out_channel=2,
        num_blocks=[4, 6, 6, 8],
        num_heads=[1, 2, 4, 8],
        channels=[48, 96, 192, 384],
        num_refinement=4,
        expansion_factor=2.66,
        num_experts=None,
        top_k=1,
        drop_path=0.0,
        norm_type=None,
        post_norm=False,
        ms_refinement=False,
        time_cond=False,
        label_cond=False,
        num_classes=9,
        timestep_scale=50.0,
        label_class_scale=10.0,
        time_embed_scale=1,
        class_dropout_prob=0.1,
        labels_embed_magnitude_scale=1.0,
    ):
        super().__init__()

        # ---- Validation ----
        assert (
            len(num_blocks) == len(num_heads) == len(channels)
        ), "num_blocks, num_heads, and channels must have the same length"
        # (Relaxed) allow L == 1 now
        self.L = len(channels)
        self.channels = channels
        self.time_cond = time_cond
        self.label_cond = label_cond
        self.num_classes = num_classes
        self.timestep_scale = timestep_scale
        self.label_class_scale = label_class_scale
        self.ms_refinement = ms_refinement
        L = self.L

        self.emb_channels = channels[0] * time_embed_scale if time_cond else None

        # Shallow embedding at level 0
        self.embed_conv = nn.Conv2d(in_channel, channels[0], kernel_size=3, padding=1, bias=False)

        # Encoders: level 0..L-1
        self.encoders = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    *[
                        TransformerBlock(
                            channels[i],
                            num_heads[i],
                            expansion_factor,
                            num_experts,
                            top_k,
                            norm_type=norm_type,
                            post_norm=post_norm,
                            emb_channels=self.emb_channels,
                        )
                        for _ in range(num_blocks[i])
                    ]
                )
                for i in range(L)
            ]
        )

        # Down path: L-1 transitions
        self.downs = nn.ModuleList([DownSample(channels[i]) for i in range(L - 1)])

        # Up path: L-1 transitions (index i corresponds to going from level i+1 -> i)
        self.ups = nn.ModuleList([UpSample(channels[i + 1]) for i in range(L - 1)])

        # Reduce 1x1 after concatenation at each decoder stage i (levels i = L-2 .. 0)
        self.reduces = nn.ModuleList(
            [nn.Conv2d(2 * channels[i], channels[i], kernel_size=1, bias=False) for i in range(L - 1)]
        )

        # Decoders: one per stage for levels i = 0..L-2
        self.decoders = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    *[
                        TransformerBlock(
                            channels[i],
                            num_heads[i],
                            expansion_factor,
                            num_experts,
                            top_k,
                            drop_path=drop_path,
                            norm_type=norm_type,
                            post_norm=post_norm,
                            emb_channels=self.emb_channels,
                        )
                        for _ in range(num_blocks[i])
                    ]
                )
                for i in range(L - 1)
            ]
        )
        self.refinement = TimestepEmbedSequential(
            *[
                TransformerBlock(
                    channels[0],
                    num_heads[0],
                    expansion_factor,
                    num_experts,
                    top_k,
                    drop_path=drop_path,
                    norm_type=norm_type,
                    post_norm=post_norm,
                    emb_channels=self.emb_channels,
                )
                for _ in range(num_refinement)
            ]
        )
        # Final conv
        self.output = nn.Conv2d(channels[0], out_channel, kernel_size=3, padding=1, bias=False)

        # For padding to be divisible by 2^(L-1)
        self.pad_factor = 2 ** (L - 1)

        if self.time_cond:
            self.time_embed = nn.Sequential(
                nn.Linear(self.channels[0], self.emb_channels),
                nn.SiLU(),
                nn.Linear(self.emb_channels, self.emb_channels),
            )

        if self.label_cond:
            # self.label_emb = LabelEmbedder(self.num_classes, self.emb_channels, class_dropout_prob, labels_embed_magnitude_scale)
            self.label_emb = nn.Sequential(
                nn.Linear(self.channels[0], self.emb_channels),
                nn.SiLU(),
                nn.Linear(self.emb_channels, self.emb_channels),
            )

    def forward(self, x, cas_skips=None, timestep=None, label=None):
        L = self.L

        # Enforce cas_skips contract if provided
        if cas_skips is not None:
            if L >= 2:
                assert len(cas_skips) == (L - 1), f"cas_skips must have length {L-1} if L >= 2, got {len(cas_skips)}"
            else:
                assert len(cas_skips) == 1, f"cas_skips must have length 1 if L = 1, got {len(cas_skips)}"

        x, padding_sizes = divisible_pad_t(x, k=self.pad_factor)

        if timestep is not None and self.time_cond:
            timesteps = torch.tensor(float(timestep) * self.timestep_scale, device=x.device, dtype=x.dtype).expand(
                x.shape[0]
            )
            t_emb = timestep_embedding(timesteps, self.channels[0], repeat_only=False)
            emb = self.time_embed(t_emb)
        else:
            emb = None

        if label is not None and self.label_cond:
            y = torch.tensor(float(label) * self.label_class_scale, device=x.device, dtype=x.dtype).expand(x.shape[0])
            y_emb = timestep_embedding(y, self.channels[0], repeat_only=False)
            emb = emb + self.label_emb(y_emb) if emb is not None else self.label_emb(y_emb)

        # ---- Encoder ----
        feats = [None] * L  # store encoder outputs at each level
        x0 = self.embed_conv(x)
        feats[0] = self.encoders[0](x0, emb=emb)

        for i in range(1, L):
            down_in = self.downs[i - 1](feats[i - 1])
            if cas_skips is not None:
                # cas_skips length is L-1, where cas_skips[i-1] matches level i
                down_in = down_in + cas_skips[i - 1]
            feats[i] = self.encoders[i](down_in, emb=emb)

        # ---- Decoder ----
        if L >= 2:
            x_dec = feats[-1]  # start from bottom level (L-1)
            dec_outs = [None] * (L - 1)  # per-level decoder outputs for i=0..L-2

            # iterate levels from L-2 down to 0
            for i in reversed(range(L - 1)):
                up = self.ups[i](x_dec)  # upsample from level i+1 -> i
                cat = torch.cat([up, feats[i]], dim=1)
                red = self.reduces[i](cat)  # project 2*C_i -> C_i
                x_dec = self.decoders[i](red, emb=emb)  # decode at level i
                dec_outs[i] = x_dec  # store decoder output aligned with level i

            ref_in = dec_outs[0]  # refinement at top level (level 0), unchanged
        else:
            # L == 1: no decoder stages; refine the single encoder output
            dec_outs = []
            ref_in = feats[0] + cas_skips[0] if cas_skips is not None else feats[0]

        # ---- Refinement + Output ----
        fr = self.refinement(ref_in, emb=emb)
        out = self.output(fr)

        out = inverse_divisible_pad_t(out, padding_sizes)

        # ---- Cascade outputs (generalized) ----
        # Return: [dec_out(level=1), ..., dec_out(level=L-2), enc_out(level=L-1)]
        cas_outs = []
        if L >= 2:
            for i in range(1, L - 1):
                cas_outs.append(dec_outs[i])  # decoder outputs for levels 1..L-2
            cas_outs.append(feats[-1])  # deepest encoder output at level L-1
        else:
            cas_outs.append(fr)  # L == 1: empty cascade outputs

        return out, tuple(cas_outs)
