"""Loki top-K masker (V81, selector ablation of the paper's component table).

Loki (Singhania et al., NeurIPS 2024, hpcgroup/loki, methods/pca_topk/utils.py
`mask_attn_pca_topk`) ranks the cached keys in a low-dimensional PCA space of the keys:

    offline   one PCA per (layer, KV head) of the POST-rotary keys of a calibration corpus
              (wikitext-2 validation, 8192-token chunks; run_v59_loki_pca.py reproduces the
              released saver + pca.py), W_r = its leading r components (columns)
    online    s_hat = (q W_r)(k W_r)^T / sqrt(D)   -- raw (uncentred) queries and keys, the
              PCA mean is never used -- and the selector is top-k on s_hat

The released code stores the keys rotated into the full PCA basis and reads their first r
coordinates; here the r coordinates are kept as a separate fp32 code (identical scores, the
key cache itself stays in the model basis so every other stage is untouched).  r = 32 is
Loki's d_f = 0.25 on the 128-dim heads of Llama-3.1-8B.  Nothing is learned or adapted to
the request.

State lives in sparse_meta_data["loki_sel"][layer_idx] (reset with the request); the
per-token score is published under "static_scores" exactly as the static / LSH selectors
do, so the cluster expansion can order the tokens of a cluster by it.
"""
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch

from sparse_attention_hub.sparse_attention.research_attention.maskers.base import (
    AttentionTensorDimensions,
    MaskerConfig,
    MaskerRegistry,
)
from sparse_attention_hub.sparse_attention.utils.kv_utils import (
    _get_num_key_value_groups,
)
from sparse_attention_hub.sparse_attention.utils.mask import Mask

from ..base import TopKMasker, TopKMaskerConfig


@dataclass
class LokiTopKMaskerConfig(TopKMaskerConfig):
    """Configuration for LokiTopKMasker."""

    pca_file: str = ""       # torch file from run_v59_loki_pca.py: comps (L,Hkv,D,D), columns = components
    rank: int = 32           # r (Loki's d_f * head_dim)
    search_space: Dict[str, Any] = field(default_factory=dict)


@MaskerRegistry.register(LokiTopKMaskerConfig)
class LokiTopKMasker(TopKMasker):
    """Top-k on Loki's rank-r key-PCA approximation of the logits."""

    def __init__(self, config: LokiTopKMaskerConfig) -> None:
        super().__init__(config)
        self.heavy_size = config.heavy_size
        self.rank = int(config.rank)
        if not config.pca_file:
            raise ValueError("pca_file is required")
        d = torch.load(config.pca_file, map_location="cpu")
        comps = d["comps"].float()                                   # (L,Hkv,D,D)
        if self.rank < 1 or self.rank > comps.shape[-1]:
            raise ValueError(f"rank must be in [1, {comps.shape[-1]}], got {self.rank}")
        self._W: torch.Tensor = comps[..., : self.rank].contiguous()  # (L,Hkv,D,r)
        self._dev: Optional[torch.device] = None

    def _to_device(self, device: torch.device) -> None:
        if self._dev != device:
            self._W = self._W.to(device)
            self._dev = device

    @torch.no_grad()
    def _extend(self, ent: Dict[str, Any], keys: torch.Tensor, W: torch.Tensor) -> Dict[str, Any]:
        """Project the keys the code cache has not seen yet: (B,Hkv,dK,D) @ (Hkv,D,r)."""
        K0, K = ent["K"], keys.shape[2]
        if K <= K0:
            return ent
        z = torch.matmul(keys[:, :, K0:, :].to(torch.float32), W.unsqueeze(0))   # (B,Hkv,dK,r)
        ent["codes"] = z if ent["codes"] is None else torch.cat([ent["codes"], z], 2)
        ent["K"] = K
        return ent

    @torch.no_grad()
    def loki_scores(self, keys: torch.Tensor, queries: torch.Tensor,
                    sparse_meta_data: Dict[str, Any], layer_idx: int) -> torch.Tensor:
        """Approximate logits (B,H,Q,K) of every key for every query row."""
        self._to_device(keys.device)
        if layer_idx >= self._W.shape[0] or keys.shape[1] != self._W.shape[1] \
                or keys.shape[-1] != self._W.shape[2]:
            raise ValueError(f"Loki PCA file does not match the model: W {tuple(self._W.shape)}, "
                             f"layer {layer_idx}, keys {tuple(keys.shape)}")
        W = self._W[layer_idx]                                                   # (Hkv,D,r)
        ng = _get_num_key_value_groups(queries, keys)
        B, H, Q, D = queries.shape
        Hkv, K = keys.shape[1], keys.shape[2]
        cache = sparse_meta_data.setdefault("loki_sel", {})
        ent = cache.get(layer_idx)
        if ent is None or K < ent["K"] or ent["codes"].shape[0] != B:
            ent = {"K": 0, "codes": None}
        ent = self._extend(ent, keys, W)
        cache[layer_idx] = ent
        # queries of the ng heads that share a KV head are projected with that head's W_r
        qw = torch.matmul(queries.to(torch.float32).reshape(B, Hkv, ng * Q, D), W.unsqueeze(0))   # (B,Hkv,ng*Q,r)
        s = torch.matmul(qw, ent["codes"].transpose(-1, -2)) / math.sqrt(D)      # (B,Hkv,ng*Q,K)
        return s.reshape(B, H, Q, K)

    def add_mask(self, keys: torch.Tensor, queries: torch.Tensor, values: torch.Tensor,
                 attention_mask: torch.Tensor, scaling: float, dropout: float,
                 sparse_meta_data: Dict[str, Any], previous_mask: Mask, **kwargs: Any) -> Mask:
        if sparse_meta_data is None:
            raise ValueError("sparse_meta_data cannot be None")
        layer_idx: Optional[int] = kwargs.get("layer_idx")
        if layer_idx is None:
            raise ValueError("layer_idx must be provided in kwargs")
        if previous_mask.is_full_mask():
            return previous_mask
        dims: AttentionTensorDimensions = self._extract_tensor_dimensions(keys, queries)
        k_eff: int = self._calculate_effective_size(self.heavy_size, dims.seq_len_keys)
        if dims.seq_len_keys <= k_eff:
            return self._create_full_mask(dims, previous_mask.dtype, previous_mask.device)
        scores = self.loki_scores(keys, queries, sparse_meta_data, layer_idx)
        K = keys.shape[2]
        if attention_mask is not None:
            scores = scores + attention_mask[:, :, :, :K].to(scores.dtype)
        # publish the per-token score (as the static selector does) so a cluster
        # expansion can order the tokens of a cluster by it
        sparse_meta_data.setdefault("static_scores", {})[layer_idx] = scores
        idx = self._get_topk_indices_from_inactive_positions(scores, previous_mask, k_eff)
        mask = self._create_mask_from_rowise_indices(dims, idx, keys.device, previous_mask.dtype)
        return previous_mask.merge_mask(mask, inplace=False)

    @classmethod
    def create_from_config(cls, config: MaskerConfig) -> "LokiTopKMasker":
        if not isinstance(config, LokiTopKMaskerConfig):
            raise ValueError(f"Invalid config type: {type(config)}")
        return cls(config)
