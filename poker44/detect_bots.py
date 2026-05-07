from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, List, Tuple
import numpy as np


_REPO_ROOT = Path(__file__).resolve().parent.parent

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

def detect_bots(
    chunks: List[List[dict[str, Any]]],
) -> Tuple[List[float], List[bool]]:
    global _scorer
    chunk_list = chunks or []
    if not chunk_list:
        return [], []

    chunk_score_matrix = [
        [_scorer.score_hand(h or {}) for h in chunk] for chunk in chunk_list
    ]

    risk_scores, predictions = [], []
    for chunk_score_row in chunk_score_matrix:
        score_row = np.array(chunk_score_row, np.float32)
        bot_mask = score_row > 0.5
        n_hand = len(score_row)
        pred = np.sum(bot_mask) / n_hand > 0.5
        score = np.mean(score_row[bot_mask]) if pred else np.mean(score_row[~bot_mask])
        risk_scores.append(score.item())
        predictions.append(pred.item())

    return risk_scores, predictions



if __name__ == "__main__":
    chunks = [
          [
            {
              "metadata": {
                "game_type": "Hold'em",
                "limit_type": "No Limit",
                "max_seats": 6,
                "hero_seat": 2,
                "hand_ended_on_street": "",
                "button_seat": 0,
                "sb": 0.01,
                "bb": 0.02,
                "ante": 0,
                "rng_seed_commitment": None
              },
              "players": [
                {
                  "player_uid": "seat_1",
                  "seat": 1,
                  "starting_stack": 0.06,
                  "hole_cards": None,
                  "showed_hand": False
                },
                {
                  "player_uid": "seat_2",
                  "seat": 2,
                  "starting_stack": 0.06,
                  "hole_cards": None,
                  "showed_hand": False
                },
                {
                  "player_uid": "seat_3",
                  "seat": 3,
                  "starting_stack": 0.06,
                  "hole_cards": None,
                  "showed_hand": False
                },
                {
                  "player_uid": "seat_4",
                  "seat": 4,
                  "starting_stack": 0.06,
                  "hole_cards": None,
                  "showed_hand": False
                },
                {
                  "player_uid": "seat_5",
                  "seat": 5,
                  "starting_stack": 0.06,
                  "hole_cards": None,
                  "showed_hand": False
                },
                {
                  "player_uid": "seat_6",
                  "seat": 6,
                  "starting_stack": 0.06,
                  "hole_cards": None,
                  "showed_hand": False
                }
              ],
              "streets": [],
              "actions": [
                {
                  "action_id": "1",
                  "street": "preflop",
                  "actor_seat": 1,
                  "action_type": "other",
                  "amount": 0,
                  "raise_to": None,
                  "call_to": None,
                  "normalized_amount_bb": 0,
                  "pot_before": 0.9,
                  "pot_after": 0.9
                },
                {
                  "action_id": "2",
                  "street": "preflop",
                  "actor_seat": 4,
                  "action_type": "other",
                  "amount": 0,
                  "raise_to": None,
                  "call_to": None,
                  "normalized_amount_bb": 0,
                  "pot_before": 0.9,
                  "pot_after": 0.9
                },
                {
                  "action_id": "3",
                  "street": "preflop",
                  "actor_seat": 4,
                  "action_type": "other",
                  "amount": 0,
                  "raise_to": None,
                  "call_to": None,
                  "normalized_amount_bb": 0,
                  "pot_before": 0.9,
                  "pot_after": 0.9
                },
                {
                  "action_id": "4",
                  "street": "preflop",
                  "actor_seat": 6,
                  "action_type": "other",
                  "amount": 0,
                  "raise_to": None,
                  "call_to": None,
                  "normalized_amount_bb": 0,
                  "pot_before": 0.9,
                  "pot_after": 0.9
                },
                {
                  "action_id": "5",
                  "street": "preflop",
                  "actor_seat": 3,
                  "action_type": "other",
                  "amount": 0,
                  "raise_to": None,
                  "call_to": None,
                  "normalized_amount_bb": 0,
                  "pot_before": 0.9,
                  "pot_after": 0.9
                },
                {
                  "action_id": "6",
                  "street": "preflop",
                  "actor_seat": 2,
                  "action_type": "fold",
                  "amount": 0,
                  "raise_to": None,
                  "call_to": None,
                  "normalized_amount_bb": 0,
                  "pot_before": 0.9,
                  "pot_after": 0.9
                },
                {
                  "action_id": "7",
                  "street": "preflop",
                  "actor_seat": 2,
                  "action_type": "fold",
                  "amount": 0,
                  "raise_to": None,
                  "call_to": None,
                  "normalized_amount_bb": 0,
                  "pot_before": 0.9,
                  "pot_after": 0.9
                },
                {
                  "action_id": "8",
                  "street": "preflop",
                  "actor_seat": 5,
                  "action_type": "fold",
                  "amount": 0,
                  "raise_to": None,
                  "call_to": None,
                  "normalized_amount_bb": 0,
                  "pot_before": 0.9,
                  "pot_after": 0.9
                },
                {
                  "action_id": "9",
                  "street": "preflop",
                  "actor_seat": 6,
                  "action_type": "fold",
                  "amount": 0,
                  "raise_to": None,
                  "call_to": None,
                  "normalized_amount_bb": 0,
                  "pot_before": 0.9,
                  "pot_after": 0.9
                },
                {
                  "action_id": "10",
                  "street": "preflop",
                  "actor_seat": 3,
                  "action_type": "fold",
                  "amount": 0,
                  "raise_to": None,
                  "call_to": None,
                  "normalized_amount_bb": 0,
                  "pot_before": 0.9,
                  "pot_after": 0.9
                },
                {
                  "action_id": "11",
                  "street": "preflop",
                  "actor_seat": 3,
                  "action_type": "fold",
                  "amount": 0,
                  "raise_to": None,
                  "call_to": None,
                  "normalized_amount_bb": 0,
                  "pot_before": 0.9,
                  "pot_after": 0.9
                },
                {
                  "action_id": "12",
                  "street": "preflop",
                  "actor_seat": 1,
                  "action_type": "fold",
                  "amount": 0,
                  "raise_to": None,
                  "call_to": None,
                  "normalized_amount_bb": 0,
                  "pot_before": 0.9,
                  "pot_after": 0.9
                }
              ],
              "outcome": {
                "winners": [],
                "payouts": {},
                "total_pot": 0,
                "rake": 0,
                "result_reason": "",
                "showdown": False
              }
            }
          ]
    ]

    risk_scores, predictions = detect_bots(chunks)
    print(f"risk_scores={risk_scores}")
    print(f"predictions={predictions}")


