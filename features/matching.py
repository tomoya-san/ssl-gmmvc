"""k-NN matching of source and target WavLM features.

Before a joint GMM in :mod:`gmm` can be fitted, each source frame has to be
paired with the target frame(s) it most resembles so the two streams can be
concatenated into aligned ``(source, target)`` examples. Matching is done in
cosine space -- the natural geometry for WavLM features -- via :func:`torch.topk`
on a full distance matrix, so it runs on whatever device the features already
live on (GPU when available).

Two policies are provided:

* :func:`asymmetric_matching` -- *source -> target* only. Every source frame is
  kept and paired with its ``k`` nearest targets, so the output preserves (and
  ``k``-fold repeats) the full source sequence.
* :func:`symmetric_matching` -- *source -> target* **and** *target -> source*,
  with duplicate pairs removed. This adds target frames that no source picked but
  that picked a source, giving more balanced coverage of both sequences.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _cosine_distance_matrix(
    source_features: torch.Tensor, target_features: torch.Tensor
) -> torch.Tensor:
    """Pairwise cosine distance between two feature sets.

    Both tensors must already be on the same device.

    Parameters
    ----------
    source_features : torch.Tensor
        Source features with shape ``(n_source, feature_dim)``.
    target_features : torch.Tensor
        Target features with shape ``(n_target, feature_dim)``.

    Returns
    -------
    distances : torch.Tensor
        Cosine distance matrix with shape ``(n_source, n_target)`` where element
        ``[i, j]`` is ``1 - cosine_similarity`` between source frame ``i`` and
        target frame ``j``.
    """
    source_normalized = F.normalize(source_features, p=2, dim=-1)
    target_normalized = F.normalize(target_features, p=2, dim=-1)
    return 1.0 - source_normalized @ target_normalized.T

def symmetric_matching(
    source_features: torch.Tensor, target_features: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match in both directions and keep the union of unique pairs.

    Pairs are collected from *source -> target* and *target -> source* ``k``-NN
    matching, then deduplicated, so a frame missed by one direction can still be
    paired through the other. The matching runs on ``source_features.device``;
    ``target_features`` is moved there if needed.

    Parameters
    ----------
    source_features : torch.Tensor
        Source features with shape ``(n_source, feature_dim)``.
    target_features : torch.Tensor
        Target features with shape ``(n_target, feature_dim)``.
    k : int
        Number of nearest neighbours to find in each direction.

    Returns
    -------
    source_matched : torch.Tensor
        Source frames from the unique pairs, shape
        ``(n_unique_pairs, feature_dim)``.
    target_matched : torch.Tensor
        Target frames aligned row-for-row with ``source_matched``, shape
        ``(n_unique_pairs, feature_dim)``.
    """
    device = source_features.device
    target_features = target_features.to(device)

    n_source = source_features.shape[0]
    n_target = target_features.shape[0]

    distances = _cosine_distance_matrix(source_features, target_features)
    _, forward_neighbours = torch.topk(distances, k=k, largest=False, dim=-1)
    # The reverse distances are just the transpose; no need to recompute.
    _, reverse_neighbours = torch.topk(distances.T, k=k, largest=False, dim=-1)

    # (source_idx, target_idx) pairs from each direction, stacked as rows.
    forward_pairs = torch.stack(
        [
            torch.arange(n_source, device=device).repeat_interleave(k),
            forward_neighbours.reshape(-1),
        ],
        dim=1,
    )
    reverse_pairs = torch.stack(
        [
            reverse_neighbours.reshape(-1),
            torch.arange(n_target, device=device).repeat_interleave(k),
        ],
        dim=1,
    )

    pairs = torch.unique(torch.cat([forward_pairs, reverse_pairs], dim=0), dim=0)

    source_matched = source_features[pairs[:, 0]]
    target_matched = target_features[pairs[:, 1]]
    return source_matched, target_matched
