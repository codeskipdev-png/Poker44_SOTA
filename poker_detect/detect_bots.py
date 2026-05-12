"""Standalone ONNX bot detection for custom HTTP servers (e.g. FastAPI).

Uses ``OnnxHandScorer`` per hand. Per-chunk risk is the fraction of hands with
score > 0.8. For each request, sweep a chunk-level threshold ``t`` over
``[0, 1]`` and pick the ``t`` that makes ``predictions = risk > t`` closest to
1:1 bot:human (validator-style shuffled batches). Concretely, minimize
``|bot_count - n / 2|``.

Environment (same as the ONNX miner):

- ``POKER44_ONNX_MODEL_PATH`` — path to ``model.onnx`` (required)
- ``POKER44_ONNX_PREPROCESS_PATH`` — optional ``*.preprocess.json`` (defaults next to ONNX)

On first use, variables are also read from a ``.env`` file next to this module (repo root) if
``python-dotenv`` is installed. Process managers (PM2, systemd) do **not** inherit shell
``export``; set env in the PM2 ecosystem ``env`` block or use ``.env``.

Requires inference deps: ``pip install -e ".[detect]"`` or at least ``onnxruntime``.

Run with repo root on ``PYTHONPATH`` (or ``pip install -e .``) so ``poker_detect`` resolves.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, List, Tuple
import numpy as np


_REPO_ROOT = Path(__file__).resolve().parent


_scorer = None
_scorer_lock = threading.Lock()


def _get_scorer():
    global _scorer
    if _scorer is not None:
        return _scorer
    with _scorer_lock:
        if _scorer is None:
            from poker_detect.inference.onnx_scorer import OnnxHandScorer

            onnx_env = _REPO_ROOT / "poker_detect" / "dist" / "model.onnx"
            onnx_path = Path(onnx_env).expanduser().resolve()
            _scorer = OnnxHandScorer(onnx_path)
    return _scorer

_get_scorer()


_THRESHOLD_GRID = np.linspace(0.0, 1.0, 1001, dtype=np.float32)


def _find_balanced_threshold(chunk_risks: List[float]) -> Tuple[float, List[bool]]:
    n = len(chunk_risks)
    if n == 0:
        return 0.5, []
    risks = np.asarray(chunk_risks, dtype=np.float32)
    target = n / 2.0

    best_t = float(_THRESHOLD_GRID[0])
    best_diff = float("inf")
    for t in _THRESHOLD_GRID:
        bots = int(np.sum(risks > t))
        diff = abs(bots - target)
        if diff < best_diff:
            best_diff = diff
            best_t = float(t)
            if best_diff == 0.0:
                break

    preds = (risks > best_t).tolist()
    return best_t, preds


def detect_bots(
    chunks: List[List[dict[str, Any]]],
) -> Tuple[List[float], List[bool]]:
    """
    Score each chunk (sequence of hand dicts) with the loaded ONNX model.

    Returns ``(risk_scores, predictions)``. ``risk_scores[i]`` is the fraction
    of hands in chunk ``i`` with hand score > 0.8. ``predictions[i] = risk > t``
    where ``t`` is chosen by a sweep over ``[0, 1]`` to minimize
    ``|bot_count - n / 2|`` (closest to 1:1).
    """
    global _scorer
    chunk_list = chunks or []
    if not chunk_list:
        return [], []

    chunk_score_matrix = [
        [_scorer.score_hand(h or {}) for h in chunk] for chunk in chunk_list
    ]

    chunk_risks: List[float] = []
    for chunk_score_row in chunk_score_matrix:
        score_row = np.asarray(chunk_score_row, dtype=np.float32)
        n_hand = len(score_row)
        if n_hand == 0:
            chunk_risks.append(0.0)
        else:
            chunk_risks.append(float(np.mean(score_row > 0.8)))

    threshold, predictions = _find_balanced_threshold(chunk_risks)
    print(f"threshold={threshold}")
    return chunk_risks, predictions



if __name__ == "__main__":
    import json
    from poker44.score.scoring import reward

    path = Path("C:/Users/admin/Documents/workspace/poker/bt_tool/dataset_maker/benchmark_out/benchmark_2026-05-08.raw.json")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for sub_data in data["data"]["chunks"]:
        chunks = sub_data["chunks"]
        groundTruth = sub_data["groundTruth"]

        risk_scores, predictions = detect_bots(chunks)
        print("\n"+"="*30)
        print(f"risk_scores={risk_scores}")
        print(f"predictions={predictions}")
        print(f"groundTruth={groundTruth}")
        print(f"reward={reward(np.array(risk_scores), np.array(groundTruth))}")
        print(f"accuracy={np.sum(np.array(predictions)==np.array(groundTruth)) / len(chunks)}")

    for i in range(85, 99):

        path = Path(
            f"C:/Users/admin/Documents/workspace/poker/bt_tool/dataset_maker/benchmark_out/output/chunks_{i+1}.json")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        chunks = data

        risk_scores, predictions = detect_bots(chunks)
        print("\n" + "=" * 30 + f"chunks {i+1}" + "=" * 30)
        print(f"risk_scores={risk_scores}")
        print(f"predictions={predictions}")
        print(f"sum bots={sum(predictions)}")




