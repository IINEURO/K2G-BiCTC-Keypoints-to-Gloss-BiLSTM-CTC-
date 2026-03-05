from __future__ import annotations

from typing import List, Sequence

import torch


def ctc_greedy_decode(
    logits: torch.Tensor,
    input_lengths: torch.Tensor,
    blank_id: int,
) -> List[List[int]]:
    """
    Args:
      logits: [B,T,C]
      input_lengths: [B]
    Returns:
      Decoded token-id sequences per sample.
    """
    if logits.ndim != 3:
        raise ValueError(f"Expected logits [B,T,C], got {tuple(logits.shape)}")

    pred = torch.argmax(logits, dim=-1)
    out: List[List[int]] = []
    for b in range(pred.shape[0]):
        t_len = int(input_lengths[b].item())
        seq = pred[b, :t_len].tolist()
        collapsed: List[int] = []
        prev = None
        for x in seq:
            if x == blank_id:
                prev = x
                continue
            if prev == x:
                continue
            collapsed.append(int(x))
            prev = x
        out.append(collapsed)
    return out


def ids_to_tokens(ids: Sequence[int], id_to_token: Sequence[str]) -> List[str]:
    toks: List[str] = []
    for i in ids:
        ii = int(i)
        if ii < 0 or ii >= len(id_to_token):
            toks.append("<OOB>")
        else:
            toks.append(id_to_token[ii])
    return toks
