#!/usr/bin/env python3
from __future__ import annotations

import argparse
import builtins
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


_ORIGINAL_PRINT = builtins.print


def print(*args: Any, **kwargs: Any) -> None:
    """Prefix all logs with local wall-clock time."""
    ts = datetime.now().astimezone().strftime("[%Y-%m-%d %H:%M:%S %Z]")
    _ORIGINAL_PRINT(ts, *args, **kwargs)


# ----------------------------- utilities -----------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device(explicit_device: Optional[str] = None) -> torch.device:
    if explicit_device is not None:
        return torch.device(explicit_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    if not hasattr(torch, dtype_name):
        raise ValueError(f"Unsupported --dtype '{dtype_name}'.")
    dtype = getattr(torch, dtype_name)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"--dtype '{dtype_name}' is not a valid torch dtype.")
    return dtype


def parse_positive_int_csv(value: str, field_name: str) -> List[int]:
    parts = [p.strip() for p in str(value).split(",")]
    out: List[int] = []
    for part in parts:
        if part == "":
            continue
        iv = int(part)
        if iv <= 0:
            raise ValueError(f"{field_name} entries must be positive integers.")
        out.append(iv)
    return sorted(set(out))


def hash_int_ids(ids: Sequence[int]) -> str:
    payload = ",".join(str(int(x)) for x in ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def lcp_and_first_divergence_position(
    generated_ids: Sequence[int],
    target_ids: Sequence[int],
) -> Tuple[int, Optional[int]]:
    match_tokens = 0
    for gen_tok, tgt_tok in zip(generated_ids, target_ids):
        if int(gen_tok) != int(tgt_tok):
            break
        match_tokens += 1
    if match_tokens >= len(target_ids):
        return int(match_tokens), None
    return int(match_tokens), int(match_tokens + 1)


@torch.no_grad()
def target_token_prob_and_rank_at_divergence(
    model,
    prompt_ids: Sequence[int],
    generated_ids: Sequence[int],
    target_ids: Sequence[int],
    first_divergence_position: Optional[int],
    device: torch.device,
) -> Tuple[Optional[float], Optional[int], int]:
    if first_divergence_position is None:
        return None, None, 0
    div_pos = int(first_divergence_position)
    if div_pos <= 0 or div_pos > len(target_ids):
        return None, None, 0

    lcp_tokens = div_pos - 1
    context_ids = list(prompt_ids) + [int(x) for x in generated_ids[:lcp_tokens]]
    if len(context_ids) == 0:
        return None, None, 0

    input_ids = torch.tensor([context_ids], dtype=torch.long, device=device)
    out = model(input_ids=input_ids, use_cache=False)
    logits = out.logits[0, -1, :]
    target_token = int(target_ids[lcp_tokens])
    target_logit = logits[target_token]
    rank = int((logits > target_logit).sum().item()) + 1
    prob = float(torch.softmax(logits, dim=-1)[target_token].item())
    return prob, rank, 1


def peak_gpu_memory_mb(device: torch.device) -> Optional[float]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    try:
        return float(torch.cuda.max_memory_allocated(device) / (1024.0 ** 2))
    except Exception:
        return None


def _flatten_for_table(value: Any, out: Dict[str, Any], prefix: str = "") -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            next_prefix = f"{prefix}.{k}" if prefix else str(k)
            _flatten_for_table(v, out, next_prefix)
        return
    if isinstance(value, list):
        out[prefix] = json.dumps(value, ensure_ascii=False)
        return
    if isinstance(value, str):
        out[prefix] = value.replace("\r", "\\r").replace("\n", "\\n")
        return
    out[prefix] = value


def write_table(path: str, rows: Sequence[Dict[str, Any]], delimiter: str = ",") -> None:
    os.makedirs(str(Path(path).parent), exist_ok=True)
    flat_rows: List[Dict[str, Any]] = []
    fieldnames: List[str] = []
    seen = set()

    for row in rows:
        flat: Dict[str, Any] = {}
        _flatten_for_table(row, flat)
        flat_rows.append(flat)
        for key in flat.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter, extrasaction="ignore")
        writer.writeheader()
        for row in flat_rows:
            writer.writerow(row)


def write_manifest(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(str(Path(path).parent), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)


def append_table_row(
    path: str,
    row: Dict[str, Any],
    fieldnames_state: Dict[str, Any],
    delimiter: str = ",",
) -> None:
    os.makedirs(str(Path(path).parent), exist_ok=True)
    flat: Dict[str, Any] = {}
    _flatten_for_table(row, flat)

    fieldnames = fieldnames_state.get("fieldnames")
    if fieldnames is None:
        fieldnames = list(flat.keys())
        fieldnames_state["fieldnames"] = fieldnames
        mode = "w"
    else:
        mode = "a"

    with open(path, mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter, extrasaction="ignore")
        if mode == "w":
            writer.writeheader()
        writer.writerow(flat)


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    # Live restart rows can contain very large JSON fields; raise CSV parser cap for resume.
    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit(2 ** 31 - 1)
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_csv_header(path: Path) -> Optional[List[str]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return None
    return list(header)


def resume_identity(row: Dict[str, Any]) -> str:
    row_uid = str(row.get("row_uid", "")).strip()
    if row_uid != "":
        return f"row_uid::{row_uid}"
    return f"pairing_key::{str(row.get('pairing_key', '')).strip()}"


def build_resume_keys(rows: Sequence[Dict[str, Any]]) -> List[str]:
    counts: Dict[str, int] = defaultdict(int)
    out: List[str] = []
    for row in rows:
        ident = resume_identity(row)
        counts[ident] += 1
        out.append(f"{ident}::occ{counts[ident]}")
    return out


def _to_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _to_int(value: Any) -> int:
    try:
        return int(float(value))
    except Exception:
        return 0


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def summarize_results_flat(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}

    n = len(rows)
    by_dup_count = Counter(_to_int(r.get("duplication_count")) for r in rows)
    by_dup_any_success = defaultdict(int)
    by_dup_objective_any_success = defaultdict(int)

    any_success_count = 0
    success_count_total = 0
    distinct_success_count_total = 0
    objective_any_success_count = 0
    objective_success_count_total = 0
    objective_distinct_success_count_total = 0
    dual_any_paraphrased_count = 0
    dual_any_original_count = 0
    dual_any_union_count = 0
    dual_success_paraphrased_total = 0
    dual_success_original_total = 0
    dual_success_both_total = 0
    dual_success_union_total = 0

    for r in rows:
        dup = _to_int(r.get("duplication_count"))
        any_success = _to_bool(r.get("ourdef_any_success"))
        objective_any_success = _to_bool(r.get("ourdef_objective_any_success"))
        dual_any_paraphrased = _to_bool(r.get("ourdef_dual_any_success_paraphrased"))
        dual_any_original = _to_bool(r.get("ourdef_dual_any_success_original"))
        dual_any_union = _to_bool(r.get("ourdef_dual_any_success_union"))
        by_dup_any_success[dup] += int(any_success)
        by_dup_objective_any_success[dup] += int(objective_any_success)
        any_success_count += int(any_success)
        objective_any_success_count += int(objective_any_success)
        dual_any_paraphrased_count += int(dual_any_paraphrased)
        dual_any_original_count += int(dual_any_original)
        dual_any_union_count += int(dual_any_union)
        success_count_total += _to_int(r.get("ourdef_success_count"))
        distinct_success_count_total += _to_int(r.get("ourdef_distinct_success_count"))
        objective_success_count_total += _to_int(r.get("ourdef_objective_success_count"))
        objective_distinct_success_count_total += _to_int(r.get("ourdef_objective_distinct_success_count"))
        dual_success_paraphrased_total += _to_int(r.get("ourdef_dual_success_count_paraphrased"))
        dual_success_original_total += _to_int(r.get("ourdef_dual_success_count_original"))
        dual_success_both_total += _to_int(r.get("ourdef_dual_success_count_both"))
        dual_success_union_total += _to_int(r.get("ourdef_dual_success_count_union"))

    return {
        "n_rows": n,
        "ourdef_any_success_count": any_success_count,
        "ourdef_any_success_rate": any_success_count / float(n),
        "ourdef_success_count_total": success_count_total,
        "ourdef_success_count_mean": success_count_total / float(n),
        "ourdef_distinct_success_count_total": distinct_success_count_total,
        "ourdef_distinct_success_count_mean": distinct_success_count_total / float(n),
        "ourdef_objective_any_success_count": objective_any_success_count,
        "ourdef_objective_any_success_rate": objective_any_success_count / float(n),
        "ourdef_objective_success_count_total": objective_success_count_total,
        "ourdef_objective_success_count_mean": objective_success_count_total / float(n),
        "ourdef_objective_distinct_success_count_total": objective_distinct_success_count_total,
        "ourdef_objective_distinct_success_count_mean": (
            objective_distinct_success_count_total / float(n)
        ),
        "ourdef_dual_any_success_paraphrased_count": dual_any_paraphrased_count,
        "ourdef_dual_any_success_paraphrased_rate": dual_any_paraphrased_count / float(n),
        "ourdef_dual_any_success_original_count": dual_any_original_count,
        "ourdef_dual_any_success_original_rate": dual_any_original_count / float(n),
        "ourdef_dual_any_success_union_count": dual_any_union_count,
        "ourdef_dual_any_success_union_rate": dual_any_union_count / float(n),
        "ourdef_dual_success_count_paraphrased_total": dual_success_paraphrased_total,
        "ourdef_dual_success_count_original_total": dual_success_original_total,
        "ourdef_dual_success_count_both_total": dual_success_both_total,
        "ourdef_dual_success_count_union_total": dual_success_union_total,
        "by_dup_count": dict(sorted(by_dup_count.items())),
        "by_dup_any_success": dict(sorted(by_dup_any_success.items())),
        "by_dup_objective_any_success": dict(sorted(by_dup_objective_any_success.items())),
    }


def summarize_compute_flat(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}

    n = float(len(rows))
    time_sec_total = sum(_to_float(r.get("compute.time_sec")) for r in rows)
    search_steps_total = sum(_to_int(r.get("compute.search_steps")) for r in rows)
    forward_calls_total = sum(_to_int(r.get("compute.forward_calls")) for r in rows)
    backward_calls_total = sum(_to_int(r.get("compute.backward_calls")) for r in rows)
    generate_calls_total = sum(_to_int(r.get("compute.generate_calls")) for r in rows)
    model_calls_total = sum(_to_int(r.get("compute.model_calls")) for r in rows)

    summary = {
        "n_targets": int(n),
        "time_sec_total": time_sec_total,
        "search_steps_total": int(search_steps_total),
        "forward_calls_total": int(forward_calls_total),
        "backward_calls_total": int(backward_calls_total),
        "generate_calls_total": int(generate_calls_total),
        "model_calls_total": int(model_calls_total),
    }
    summary["time_sec_mean"] = summary["time_sec_total"] / n
    summary["model_calls_mean"] = summary["model_calls_total"] / n
    return summary


def parse_token_ids(value: str) -> List[int]:
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise ValueError("target_token_ids must parse to a list.")
    return [int(x) for x in parsed]


def parse_optional_token_ids(value: Any) -> Optional[List[int]]:
    if value is None:
        return None
    raw = str(value).strip()
    if raw == "":
        return None
    try:
        return parse_token_ids(raw)
    except Exception:
        return None


def parse_dup_caps(spec: str) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"Invalid dup cap '{chunk}'. Expected format 'dup:count'.")
        d, c = chunk.split(":", 1)
        dup = int(d.strip())
        cap = int(c.strip())
        if cap < 0:
            raise ValueError(f"Dup cap must be >= 0. Got {dup}:{cap}")
        out[dup] = cap
    if not out:
        raise ValueError("No dup caps parsed. Provide --dup_caps like '0:100,1:100,4:100,16:100,64:100,256:100'.")
    return out


def select_targets_by_dup(
    rows: Sequence[Dict[str, Any]],
    dup_caps: Dict[int, int],
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    by_dup: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        dup = int(r["duplication_count"])
        by_dup[dup].append(r)

    selected: List[Dict[str, Any]] = []
    diagnostics: List[Dict[str, Any]] = []

    for dup in sorted(dup_caps.keys(), reverse=True):
        bucket = sorted(by_dup.get(dup, []), key=lambda x: str(x.get("pairing_key", "")))
        requested = int(dup_caps[dup])
        available = len(bucket)
        if requested <= 0:
            diagnostics.append(
                {
                    "duplication_count": dup,
                    "available": available,
                    "requested": requested,
                    "selected": 0,
                }
            )
            continue

        n_take = min(requested, available)
        if n_take == available:
            chosen = list(bucket)
        else:
            rng = random.Random(seed + dup * 1009)
            chosen = rng.sample(bucket, n_take)
            chosen = sorted(chosen, key=lambda x: str(x.get("pairing_key", "")))

        selected.extend(chosen)
        diagnostics.append(
            {
                "duplication_count": dup,
                "available": available,
                "requested": requested,
                "selected": n_take,
            }
        )

    selected = sorted(
        selected,
        key=lambda x: (
            -int(x["duplication_count"]),
            str(x.get("dataset_name", "")),
            str(x.get("dataset_config", "")),
            str(x.get("source_split", "")),
            str(x.get("source_id", "")),
            int(x.get("target_len", 0)),
            str(x.get("span_slot", "")),
            int(x.get("target_start_token_idx", 0)),
            int(x.get("target_end_token_idx", 0)),
            str(x.get("pairing_key", "")),
        ),
    )
    return selected, diagnostics


# ----------------------------- optimization -----------------------------


def optimize_gcg_prefix(
    model,
    tokenizer,
    embedding_matrix: torch.Tensor,
    scaffold_ids: Sequence[int],
    target_ids: Sequence[int],
    adv_prefix_len: int,
    num_steps: int,
    no_improve_patience: int,
    topk: int,
    batch_size: int,
    mini_batch_size: int,
    device: torch.device,
    generation_success_cache: Optional[
        Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], Tuple[bool, List[int]]]
    ] = None,
    position_sampling: str = "grad",
    dedupe_candidates: bool = True,
    adaptive_schedule: bool = True,
    num_mutations_per_candidate: int = 1,
    random_candidate_frac: float = 0.0,
    topk_final: Optional[int] = None,
    no_improve_patience_final: Optional[int] = None,
    sa_accept_worse: bool = False,
    sa_temp_init: float = 0.05,
    sa_temp_final: float = 0.005,
    log_gcg_iterations: bool = False,
    gcg_log_every: int = 25,
    on_new_best: Optional[Callable[[Dict[str, Any]], None]] = None,
    on_step_end: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    if adv_prefix_len <= 0:
        raise ValueError("adv_prefix_len must be positive.")
    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")
    if no_improve_patience <= 0:
        raise ValueError("no_improve_patience must be positive.")
    if gcg_log_every <= 0:
        raise ValueError("gcg_log_every must be positive.")
    if position_sampling not in {"uniform", "grad"}:
        raise ValueError("position_sampling must be one of {'uniform', 'grad'}.")
    if num_mutations_per_candidate <= 0:
        raise ValueError("num_mutations_per_candidate must be positive.")
    if random_candidate_frac < 0.0 or random_candidate_frac >= 1.0:
        raise ValueError("random_candidate_frac must be in [0.0, 1.0).")
    if topk_final is not None and topk_final <= 0:
        raise ValueError("topk_final must be positive when provided.")
    if no_improve_patience_final is not None and no_improve_patience_final <= 0:
        raise ValueError("no_improve_patience_final must be positive when provided.")
    if sa_temp_init <= 0:
        raise ValueError("sa_temp_init must be positive.")
    if sa_temp_final <= 0:
        raise ValueError("sa_temp_final must be positive.")
    if generation_success_cache is None:
        generation_success_cache = {}

    vocab_size = embedding_matrix.shape[0]
    scaffold = torch.tensor(list(scaffold_ids), dtype=torch.long, device=device)
    target = torch.tensor(list(target_ids), dtype=torch.long, device=device)
    target_list = target.tolist()
    free_tokens = torch.randint(0, vocab_size, (adv_prefix_len,), device=device)

    prompt_len = scaffold.shape[0] + adv_prefix_len
    input_ids = torch.cat([scaffold, free_tokens, target], dim=0).long()

    free_slice = slice(scaffold.shape[0], prompt_len)
    target_slice = slice(prompt_len, prompt_len + target.shape[0])
    loss_slice = slice(prompt_len - 1, prompt_len + target.shape[0] - 1)

    best_loss = float("inf")
    best_free = input_ids[free_slice].clone()
    initial_free = input_ids[free_slice].clone()
    initial_loss: Optional[float] = None
    best_step_idx = 0
    accepted_updates = 0
    control_token_changes_total = 0

    forward_calls = 0
    backward_calls = 0
    generate_calls = 0
    search_steps = 0
    no_improve_steps = 0
    success_found = False
    success_step: Optional[int] = None
    success_prompt_ids: Optional[torch.Tensor] = None
    best_prompt_generation_success = False
    best_prompt_generation_ids: List[int] = []
    stop_reason = "max_steps"

    def _emit_step_event(
        step_idx_1b: int,
        current_loss_value: float,
        best_loss_value: float,
        no_improve_steps_value: int,
        effective_patience_value: int,
        n_candidates_value: int,
        local_improved_value: bool,
        global_improved_value: bool,
        accepted_update_value: bool,
        early_stop_triggered_value: bool,
        stop_reason_value: Optional[str] = None,
    ) -> None:
        if on_step_end is None:
            return
        current_free_ids = [int(x) for x in input_ids[free_slice].tolist()]
        best_free_ids = [int(x) for x in best_free.tolist()]
        current_prompt_ids = [int(x) for x in torch.cat([scaffold, input_ids[free_slice]], dim=0).tolist()]
        best_prompt_ids_evt = [int(x) for x in torch.cat([scaffold, best_free], dim=0).tolist()]
        on_step_end(
            {
                "step_idx": int(step_idx_1b),
                "current_loss": float(current_loss_value),
                "best_loss": float(best_loss_value),
                "loss_initial": (float(initial_loss) if initial_loss is not None else None),
                "best_step_idx": int(best_step_idx),
                "current_prompt_ids": current_prompt_ids,
                "best_prompt_ids": best_prompt_ids_evt,
                "initial_free_ids": [int(x) for x in initial_free.tolist()],
                "current_free_ids": current_free_ids,
                "best_free_ids": best_free_ids,
                "no_improve_steps": int(no_improve_steps_value),
                "effective_patience": int(effective_patience_value),
                "n_candidates": int(n_candidates_value),
                "local_improved": bool(local_improved_value),
                "global_improved": bool(global_improved_value),
                "accepted_update": bool(accepted_update_value),
                "num_accepted_updates": int(accepted_updates),
                "num_control_token_changes": int(control_token_changes_total),
                "early_stop_triggered": bool(early_stop_triggered_value),
                "stop_reason": stop_reason_value,
            }
        )

    for step_idx in range(num_steps):
        search_steps += 1
        one_hot = F.one_hot(input_ids, vocab_size).to(dtype=embedding_matrix.dtype).unsqueeze(0)
        one_hot.requires_grad_(True)
        inputs_embeds = torch.matmul(one_hot, embedding_matrix)

        outputs = model(inputs_embeds=inputs_embeds, use_cache=False)
        forward_calls += 1
        loss = F.cross_entropy(outputs.logits[0, loss_slice, :], input_ids[target_slice])
        current_loss = float(loss.item())
        if initial_loss is None:
            initial_loss = float(current_loss)

        grad = torch.autograd.grad(loss, one_hot)[0][:, free_slice]
        backward_calls += 1

        with torch.no_grad():
            progress = step_idx / max(1, num_steps - 1)
            if adaptive_schedule:
                k_start = min(topk, vocab_size)
                k_end_raw = topk_final if topk_final is not None else max(1, k_start // 2)
                k_end = max(1, min(k_start, k_end_raw, vocab_size))
                k = max(1, int(round(k_start + (k_end - k_start) * progress)))
                patience_start = no_improve_patience
                patience_end_raw = (
                    no_improve_patience_final
                    if no_improve_patience_final is not None
                    else max(1, patience_start // 2)
                )
                patience_end = max(1, min(patience_start, patience_end_raw))
                effective_patience = max(
                    1, int(round(patience_start + (patience_end - patience_start) * progress))
                )
            else:
                k = min(topk, vocab_size)
                effective_patience = no_improve_patience

            top_indices = torch.topk(-grad[0], k=k, dim=1).indices  # [adv_prefix_len, k]

            free_token_ids = one_hot[0, free_slice].argmax(-1)
            n_grad = batch_size - int(round(batch_size * random_candidate_frac))
            n_grad = max(0, min(batch_size, n_grad))
            n_rand = batch_size - n_grad
            n_mut = max(1, min(adv_prefix_len, int(num_mutations_per_candidate)))

            candidate_batches: List[torch.Tensor] = []
            if n_grad > 0:
                grad_batch = free_token_ids.repeat(n_grad, 1)
                for _ in range(n_mut):
                    if position_sampling == "grad":
                        position_scores = torch.clamp(torch.amax(-grad[0], dim=1), min=0.0)
                        if float(position_scores.sum().item()) > 0:
                            new_token_loc = torch.multinomial(position_scores, num_samples=n_grad, replacement=True)
                        else:
                            new_token_loc = torch.randint(0, adv_prefix_len, (n_grad,), device=device)
                    else:
                        new_token_loc = torch.randint(0, adv_prefix_len, (n_grad,), device=device)
                    new_token_k = torch.randint(0, k, (n_grad,), device=device)
                    new_token_vals = top_indices[new_token_loc, new_token_k]
                    grad_batch[torch.arange(n_grad, device=device), new_token_loc] = new_token_vals
                candidate_batches.append(grad_batch)

            if n_rand > 0:
                rand_batch = free_token_ids.repeat(n_rand, 1)
                for _ in range(n_mut):
                    rand_loc = torch.randint(0, adv_prefix_len, (n_rand,), device=device)
                    rand_vals = torch.randint(0, vocab_size, (n_rand,), device=device)
                    rand_batch[torch.arange(n_rand, device=device), rand_loc] = rand_vals
                candidate_batches.append(rand_batch)

            free_tokens_batch = (
                torch.cat(candidate_batches, dim=0)
                if candidate_batches
                else free_token_ids.repeat(1, 1)
            )

            changed_mask = (free_tokens_batch != free_token_ids.unsqueeze(0)).any(dim=1)
            free_tokens_batch = free_tokens_batch[changed_mask]
            if dedupe_candidates and free_tokens_batch.shape[0] > 0:
                free_tokens_batch = torch.unique(free_tokens_batch, dim=0)

            if free_tokens_batch.shape[0] == 0:
                step_best_loss = current_loss
                global_improved = False
                local_improved = False
                no_improve_steps += 1
                if no_improve_steps >= effective_patience:
                    stop_reason = "no_global_improve_patience"
                    _emit_step_event(
                        step_idx_1b=step_idx + 1,
                        current_loss_value=float(current_loss),
                        best_loss_value=float(best_loss),
                        no_improve_steps_value=no_improve_steps,
                        effective_patience_value=effective_patience,
                        n_candidates_value=0,
                        local_improved_value=local_improved,
                        global_improved_value=global_improved,
                        accepted_update_value=False,
                        early_stop_triggered_value=True,
                        stop_reason_value=stop_reason,
                    )
                    if log_gcg_iterations:
                        print(
                            f"[gcg_stop] iter={step_idx + 1} reason=no_global_improve_patience "
                            f"best_loss={best_loss:.6f} k={k} patience={effective_patience}",
                            flush=True,
                        )
                    break
                _emit_step_event(
                    step_idx_1b=step_idx + 1,
                    current_loss_value=float(current_loss),
                    best_loss_value=float(best_loss),
                    no_improve_steps_value=no_improve_steps,
                    effective_patience_value=effective_patience,
                    n_candidates_value=0,
                    local_improved_value=local_improved,
                    global_improved_value=global_improved,
                    accepted_update_value=False,
                    early_stop_triggered_value=False,
                )
                continue

            n_candidates = int(free_tokens_batch.shape[0])
            candidates_input_ids = input_ids.repeat(n_candidates, 1)
            candidates_input_ids[:, free_slice] = free_tokens_batch

            cand_losses = torch.empty(n_candidates, device=device)
            for start in range(0, n_candidates, mini_batch_size):
                mb = candidates_input_ids[start:start + mini_batch_size]
                out = model(input_ids=mb, use_cache=False)
                forward_calls += 1
                labels = target.unsqueeze(0).repeat(out.logits.shape[0], 1)
                mb_loss = F.cross_entropy(
                    out.logits[:, loss_slice, :].transpose(1, 2),
                    labels,
                    reduction="none",
                ).mean(dim=1)
                cand_losses[start:start + mb.shape[0]] = mb_loss

            best_idx = int(torch.argmin(cand_losses).item())
            proposed_input_ids = candidates_input_ids[best_idx].clone().detach()
            step_best_loss = float(cand_losses[best_idx].item())
            temp = sa_temp_init + (sa_temp_final - sa_temp_init) * progress
            delta = step_best_loss - current_loss
            accepted = True
            prev_free_ids = input_ids[free_slice].clone()
            if sa_accept_worse and delta > 0:
                accept_p = math.exp(-delta / max(temp, 1e-12))
                accepted = random.random() < accept_p
            if accepted:
                input_ids = proposed_input_ids
            else:
                step_best_loss = current_loss
            current_free_after = input_ids[free_slice]
            changed_tokens = int((current_free_after != prev_free_ids).sum().item()) if accepted else 0
            accepted_update = bool(accepted and changed_tokens > 0)
            if accepted_update:
                accepted_updates += 1
                control_token_changes_total += changed_tokens
            global_improved = step_best_loss < best_loss
            local_improved = step_best_loss < current_loss
            if global_improved:
                best_loss = step_best_loss
                best_free = input_ids[free_slice].clone()
                best_step_idx = int(step_idx + 1)
                no_improve_steps = 0
                candidate_free = input_ids[free_slice].clone()
                candidate_prompt_ids = torch.cat([scaffold, candidate_free], dim=0).long().tolist()
                success, generated_ids, used_generate_call = greedy_exact_match_cached(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_ids=candidate_prompt_ids,
                    target_ids=target_list,
                    device=device,
                    generation_success_cache=generation_success_cache,
                )
                generate_calls += int(used_generate_call)
                best_prompt_generation_success = bool(success)
                best_prompt_generation_ids = [int(x) for x in generated_ids]
                if on_new_best is not None:
                    on_new_best(
                        {
                            "step_idx": int(step_idx + 1),
                            "best_loss": float(best_loss),
                            "prompt_ids": [int(x) for x in candidate_prompt_ids],
                            "generated_ids": [int(x) for x in generated_ids],
                            "success": bool(success),
                        }
                    )
                if success:
                    success_found = True
                    success_step = step_idx + 1
                    success_prompt_ids = torch.tensor(candidate_prompt_ids, dtype=torch.long, device=device)
                    stop_reason = "success"
                    _emit_step_event(
                        step_idx_1b=step_idx + 1,
                        current_loss_value=float(step_best_loss),
                        best_loss_value=float(best_loss),
                        no_improve_steps_value=no_improve_steps,
                        effective_patience_value=effective_patience,
                        n_candidates_value=n_candidates,
                        local_improved_value=local_improved,
                        global_improved_value=global_improved,
                        accepted_update_value=accepted_update,
                        early_stop_triggered_value=True,
                        stop_reason_value=stop_reason,
                    )
                    if log_gcg_iterations:
                        print(
                            f"[gcg_stop] iter={success_step} reason=success best_loss={best_loss:.6f}",
                            flush=True,
                        )
                    break
            else:
                no_improve_steps += 1
            hit_no_global_patience = no_improve_steps >= effective_patience

            if hit_no_global_patience:
                stop_reason = "no_global_improve_patience"
                _emit_step_event(
                    step_idx_1b=step_idx + 1,
                    current_loss_value=float(step_best_loss),
                    best_loss_value=float(best_loss),
                    no_improve_steps_value=no_improve_steps,
                    effective_patience_value=effective_patience,
                    n_candidates_value=n_candidates,
                    local_improved_value=local_improved,
                    global_improved_value=global_improved,
                    accepted_update_value=accepted_update,
                    early_stop_triggered_value=True,
                    stop_reason_value=stop_reason,
                )
                if log_gcg_iterations:
                    print(
                        f"[gcg_stop] iter={step_idx + 1} reason=no_global_improve_patience "
                        f"best_loss={best_loss:.6f} k={k} patience={effective_patience}",
                        flush=True,
                    )
                break
            if log_gcg_iterations and (
                step_idx == 0
                or (step_idx + 1) % gcg_log_every == 0
                or step_idx + 1 == num_steps
            ):
                print(
                    f"[gcg_iter] iter={step_idx + 1}/{num_steps} "
                    f"step_best_loss={step_best_loss:.6f} "
                    f"best_loss={best_loss:.6f} "
                    f"local_improved={int(local_improved)} global_improved={int(global_improved)} "
                    f"k={k} patience={effective_patience} candidates={n_candidates} "
                    f"accepted={int(accepted)} curr_loss={current_loss:.6f} temp={temp:.6f}",
                    flush=True,
                )
            _emit_step_event(
                step_idx_1b=step_idx + 1,
                current_loss_value=float(step_best_loss),
                best_loss_value=float(best_loss),
                no_improve_steps_value=no_improve_steps,
                effective_patience_value=effective_patience,
                n_candidates_value=n_candidates,
                local_improved_value=local_improved,
                global_improved_value=global_improved,
                accepted_update_value=accepted_update,
                early_stop_triggered_value=False,
            )

    best_prompt_ids = (
        success_prompt_ids
        if success_prompt_ids is not None
        else torch.cat([scaffold, best_free], dim=0).long()
    )
    best_prompt_list = [int(x) for x in best_prompt_ids.tolist()]
    if len(best_prompt_generation_ids) == 0:
        best_prompt_success, best_prompt_generation_ids, used_generate_call = greedy_exact_match_cached(
            model=model,
            tokenizer=tokenizer,
            prompt_ids=best_prompt_list,
            target_ids=target_list,
            device=device,
            generation_success_cache=generation_success_cache,
        )
        generate_calls += int(used_generate_call)
        best_prompt_generation_success = bool(best_prompt_success)
        if on_new_best is not None:
            on_new_best(
                {
                    "step_idx": 0,
                    "best_loss": float(best_loss),
                    "prompt_ids": [int(x) for x in best_prompt_list],
                    "generated_ids": [int(x) for x in best_prompt_generation_ids],
                    "success": bool(best_prompt_success),
                    "source": "final_fallback",
                }
            )
    else:
        best_prompt_success = bool(best_prompt_generation_success)

    return {
        "best_prompt_ids": best_prompt_ids,
        "best_loss": best_loss,
        "initial_loss": (float(initial_loss) if initial_loss is not None else None),
        "best_step_idx": int(best_step_idx),
        "initial_free_ids": [int(x) for x in initial_free.tolist()],
        "best_free_ids": [int(x) for x in best_free.tolist()],
        "num_accepted_updates": int(accepted_updates),
        "num_control_token_changes": int(control_token_changes_total),
        "final_no_improve_steps": int(no_improve_steps),
        "success_found": success_found,
        "success_step": success_step,
        "stop_reason": stop_reason,
        "search_steps": search_steps,
        "forward_calls": forward_calls,
        "backward_calls": backward_calls,
        "generate_calls": generate_calls,
        "best_prompt_generation_ids": best_prompt_generation_ids,
        "best_prompt_generation_success": best_prompt_success,
    }


def optimize_random_search_prefix(
    model,
    tokenizer,
    embedding_matrix: torch.Tensor,
    scaffold_ids: Sequence[int],
    target_ids: Sequence[int],
    adv_prefix_len: int,
    num_steps: int,
    no_improve_patience: int,
    topk: int,
    batch_size: int,
    mini_batch_size: int,
    device: torch.device,
    generation_success_cache: Optional[
        Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], Tuple[bool, List[int]]]
    ] = None,
    position_sampling: str = "grad",
    dedupe_candidates: bool = True,
    adaptive_schedule: bool = True,
    num_mutations_per_candidate: int = 1,
    random_candidate_frac: float = 0.0,
    topk_final: Optional[int] = None,
    no_improve_patience_final: Optional[int] = None,
    sa_accept_worse: bool = False,
    sa_temp_init: float = 0.05,
    sa_temp_final: float = 0.005,
    log_gcg_iterations: bool = False,
    gcg_log_every: int = 25,
    on_new_best: Optional[Callable[[Dict[str, Any]], None]] = None,
    on_step_end: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    del topk
    del position_sampling
    del adaptive_schedule
    del random_candidate_frac
    del topk_final
    del no_improve_patience_final
    del sa_accept_worse
    del sa_temp_init
    del sa_temp_final
    if adv_prefix_len <= 0:
        raise ValueError("adv_prefix_len must be positive.")
    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")
    if no_improve_patience <= 0:
        raise ValueError("no_improve_patience must be positive.")
    if batch_size <= 0 or mini_batch_size <= 0:
        raise ValueError("batch_size and mini_batch_size must be positive.")
    if gcg_log_every <= 0:
        raise ValueError("gcg_log_every must be positive.")
    if num_mutations_per_candidate <= 0:
        raise ValueError("num_mutations_per_candidate must be positive.")
    if generation_success_cache is None:
        generation_success_cache = {}

    vocab_size = embedding_matrix.shape[0]
    scaffold = torch.tensor(list(scaffold_ids), dtype=torch.long, device=device)
    target = torch.tensor(list(target_ids), dtype=torch.long, device=device)
    target_list = target.tolist()
    free_tokens = torch.randint(0, vocab_size, (adv_prefix_len,), device=device)

    prompt_len = scaffold.shape[0] + adv_prefix_len
    input_ids = torch.cat([scaffold, free_tokens, target], dim=0).long()

    free_slice = slice(scaffold.shape[0], prompt_len)
    target_slice = slice(prompt_len, prompt_len + target.shape[0])
    loss_slice = slice(prompt_len - 1, prompt_len + target.shape[0] - 1)

    best_loss = float("inf")
    best_free = input_ids[free_slice].clone()
    initial_free = input_ids[free_slice].clone()
    forward_calls = 0
    backward_calls = 0
    generate_calls = 0
    search_steps = 0
    no_improve_steps = 0
    success_found = False
    success_step: Optional[int] = None
    success_prompt_ids: Optional[torch.Tensor] = None
    best_prompt_generation_success = False
    best_prompt_generation_ids: List[int] = []
    stop_reason = "max_steps"
    initial_loss: Optional[float] = None
    best_step_idx = 0
    accepted_updates = 0
    control_token_changes_total = 0

    def _emit_step_event(
        step_idx_1b: int,
        current_loss_value: float,
        best_loss_value: float,
        no_improve_steps_value: int,
        n_candidates_value: int,
        local_improved_value: bool,
        global_improved_value: bool,
        accepted_update_value: bool,
        early_stop_triggered_value: bool,
        stop_reason_value: Optional[str] = None,
    ) -> None:
        if on_step_end is None:
            return
        current_free_ids = [int(x) for x in input_ids[free_slice].tolist()]
        best_free_ids = [int(x) for x in best_free.tolist()]
        current_prompt_ids = [int(x) for x in torch.cat([scaffold, input_ids[free_slice]], dim=0).tolist()]
        best_prompt_ids_evt = [int(x) for x in torch.cat([scaffold, best_free], dim=0).tolist()]
        on_step_end(
            {
                "step_idx": int(step_idx_1b),
                "current_loss": float(current_loss_value),
                "best_loss": float(best_loss_value),
                "loss_initial": (float(initial_loss) if initial_loss is not None else None),
                "best_step_idx": int(best_step_idx),
                "current_prompt_ids": current_prompt_ids,
                "best_prompt_ids": best_prompt_ids_evt,
                "initial_free_ids": [int(x) for x in initial_free.tolist()],
                "current_free_ids": current_free_ids,
                "best_free_ids": best_free_ids,
                "no_improve_steps": int(no_improve_steps_value),
                "effective_patience": int(no_improve_patience),
                "n_candidates": int(n_candidates_value),
                "local_improved": bool(local_improved_value),
                "global_improved": bool(global_improved_value),
                "accepted_update": bool(accepted_update_value),
                "num_accepted_updates": int(accepted_updates),
                "num_control_token_changes": int(control_token_changes_total),
                "early_stop_triggered": bool(early_stop_triggered_value),
                "stop_reason": stop_reason_value,
            }
        )

    with torch.no_grad():
        base_out = model(input_ids=input_ids.unsqueeze(0), use_cache=False)
        forward_calls += 1
        base_labels = target.unsqueeze(0)
        base_loss = F.cross_entropy(
            base_out.logits[:, loss_slice, :].transpose(1, 2),
            base_labels,
            reduction="none",
        ).mean(dim=1)[0]
        current_loss = float(base_loss.item())
        initial_loss = float(current_loss)
        best_loss = current_loss
        best_free = input_ids[free_slice].clone()

    for step_idx in range(num_steps):
        search_steps += 1
        with torch.no_grad():
            free_token_ids = input_ids[free_slice]
            n_mut = max(1, min(adv_prefix_len, int(num_mutations_per_candidate)))

            free_tokens_batch = free_token_ids.repeat(batch_size, 1)
            for _ in range(n_mut):
                new_token_loc = torch.randint(0, adv_prefix_len, (batch_size,), device=device)
                new_token_vals = torch.randint(0, vocab_size, (batch_size,), device=device)
                free_tokens_batch[torch.arange(batch_size, device=device), new_token_loc] = new_token_vals

            changed_mask = (free_tokens_batch != free_token_ids.unsqueeze(0)).any(dim=1)
            free_tokens_batch = free_tokens_batch[changed_mask]
            if dedupe_candidates and free_tokens_batch.shape[0] > 0:
                free_tokens_batch = torch.unique(free_tokens_batch, dim=0)

            if free_tokens_batch.shape[0] == 0:
                no_improve_steps += 1
                if no_improve_steps >= no_improve_patience:
                    stop_reason = "no_global_improve_patience"
                    _emit_step_event(
                        step_idx_1b=step_idx + 1,
                        current_loss_value=float(current_loss),
                        best_loss_value=float(best_loss),
                        no_improve_steps_value=no_improve_steps,
                        n_candidates_value=0,
                        local_improved_value=False,
                        global_improved_value=False,
                        accepted_update_value=False,
                        early_stop_triggered_value=True,
                        stop_reason_value=stop_reason,
                    )
                    if log_gcg_iterations:
                        print(
                            f"[random_stop] iter={step_idx + 1} reason=no_global_improve_patience "
                            f"best_loss={best_loss:.6f} patience={no_improve_patience}",
                            flush=True,
                        )
                    break
                _emit_step_event(
                    step_idx_1b=step_idx + 1,
                    current_loss_value=float(current_loss),
                    best_loss_value=float(best_loss),
                    no_improve_steps_value=no_improve_steps,
                    n_candidates_value=0,
                    local_improved_value=False,
                    global_improved_value=False,
                    accepted_update_value=False,
                    early_stop_triggered_value=False,
                )
                continue

            n_candidates = int(free_tokens_batch.shape[0])
            candidates_input_ids = input_ids.repeat(n_candidates, 1)
            candidates_input_ids[:, free_slice] = free_tokens_batch

            cand_losses = torch.empty(n_candidates, device=device)
            for start in range(0, n_candidates, mini_batch_size):
                mb = candidates_input_ids[start:start + mini_batch_size]
                out = model(input_ids=mb, use_cache=False)
                forward_calls += 1
                labels = target.unsqueeze(0).repeat(out.logits.shape[0], 1)
                mb_loss = F.cross_entropy(
                    out.logits[:, loss_slice, :].transpose(1, 2),
                    labels,
                    reduction="none",
                ).mean(dim=1)
                cand_losses[start:start + mb.shape[0]] = mb_loss

            best_idx = int(torch.argmin(cand_losses).item())
            proposed_input_ids = candidates_input_ids[best_idx].clone().detach()
            step_best_loss = float(cand_losses[best_idx].item())

            prev_free_ids = input_ids[free_slice].clone()
            local_improved = step_best_loss < current_loss
            if local_improved:
                input_ids = proposed_input_ids
                current_loss = step_best_loss
            current_free_after = input_ids[free_slice]
            changed_tokens = int((current_free_after != prev_free_ids).sum().item()) if local_improved else 0
            accepted_update = bool(local_improved and changed_tokens > 0)
            if accepted_update:
                accepted_updates += 1
                control_token_changes_total += changed_tokens

            global_improved = local_improved and current_loss < best_loss
            if global_improved:
                best_loss = current_loss
                best_free = input_ids[free_slice].clone()
                best_step_idx = int(step_idx + 1)
                no_improve_steps = 0
                candidate_free = input_ids[free_slice].clone()
                candidate_prompt_ids = torch.cat([scaffold, candidate_free], dim=0).long().tolist()
                success, generated_ids, used_generate_call = greedy_exact_match_cached(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_ids=candidate_prompt_ids,
                    target_ids=target_list,
                    device=device,
                    generation_success_cache=generation_success_cache,
                )
                generate_calls += int(used_generate_call)
                best_prompt_generation_success = bool(success)
                best_prompt_generation_ids = [int(x) for x in generated_ids]
                if on_new_best is not None:
                    on_new_best(
                        {
                            "step_idx": int(step_idx + 1),
                            "best_loss": float(best_loss),
                            "prompt_ids": [int(x) for x in candidate_prompt_ids],
                            "generated_ids": [int(x) for x in generated_ids],
                            "success": bool(success),
                        }
                    )
                if success:
                    success_found = True
                    success_step = step_idx + 1
                    success_prompt_ids = torch.tensor(candidate_prompt_ids, dtype=torch.long, device=device)
                    stop_reason = "success"
                    _emit_step_event(
                        step_idx_1b=step_idx + 1,
                        current_loss_value=float(current_loss),
                        best_loss_value=float(best_loss),
                        no_improve_steps_value=no_improve_steps,
                        n_candidates_value=n_candidates,
                        local_improved_value=local_improved,
                        global_improved_value=global_improved,
                        accepted_update_value=accepted_update,
                        early_stop_triggered_value=True,
                        stop_reason_value=stop_reason,
                    )
                    if log_gcg_iterations:
                        print(
                            f"[random_stop] iter={success_step} reason=success best_loss={best_loss:.6f}",
                            flush=True,
                        )
                    break
            else:
                no_improve_steps += 1

            if no_improve_steps >= no_improve_patience:
                stop_reason = "no_global_improve_patience"
                _emit_step_event(
                    step_idx_1b=step_idx + 1,
                    current_loss_value=float(current_loss),
                    best_loss_value=float(best_loss),
                    no_improve_steps_value=no_improve_steps,
                    n_candidates_value=n_candidates,
                    local_improved_value=local_improved,
                    global_improved_value=global_improved,
                    accepted_update_value=accepted_update,
                    early_stop_triggered_value=True,
                    stop_reason_value=stop_reason,
                )
                if log_gcg_iterations:
                    print(
                        f"[random_stop] iter={step_idx + 1} reason=no_global_improve_patience "
                        f"best_loss={best_loss:.6f} patience={no_improve_patience}",
                        flush=True,
                    )
                break

            if log_gcg_iterations and (
                step_idx == 0
                or (step_idx + 1) % gcg_log_every == 0
                or step_idx + 1 == num_steps
            ):
                print(
                    f"[random_iter] iter={step_idx + 1}/{num_steps} "
                    f"step_best_loss={step_best_loss:.6f} "
                    f"best_loss={best_loss:.6f} "
                    f"local_improved={int(local_improved)} global_improved={int(global_improved)} "
                    f"patience={no_improve_patience} candidates={n_candidates} "
                    f"curr_loss={current_loss:.6f}",
                    flush=True,
                )
            _emit_step_event(
                step_idx_1b=step_idx + 1,
                current_loss_value=float(current_loss),
                best_loss_value=float(best_loss),
                no_improve_steps_value=no_improve_steps,
                n_candidates_value=n_candidates,
                local_improved_value=local_improved,
                global_improved_value=global_improved,
                accepted_update_value=accepted_update,
                early_stop_triggered_value=False,
            )

    best_prompt_ids = (
        success_prompt_ids
        if success_prompt_ids is not None
        else torch.cat([scaffold, best_free], dim=0).long()
    )
    best_prompt_list = [int(x) for x in best_prompt_ids.tolist()]
    if len(best_prompt_generation_ids) == 0:
        best_prompt_success, best_prompt_generation_ids, used_generate_call = greedy_exact_match_cached(
            model=model,
            tokenizer=tokenizer,
            prompt_ids=best_prompt_list,
            target_ids=target_list,
            device=device,
            generation_success_cache=generation_success_cache,
        )
        generate_calls += int(used_generate_call)
        best_prompt_generation_success = bool(best_prompt_success)
        if on_new_best is not None:
            on_new_best(
                {
                    "step_idx": 0,
                    "best_loss": float(best_loss),
                    "prompt_ids": [int(x) for x in best_prompt_list],
                    "generated_ids": [int(x) for x in best_prompt_generation_ids],
                    "success": bool(best_prompt_success),
                    "source": "final_fallback",
                }
            )
    else:
        best_prompt_success = bool(best_prompt_generation_success)

    return {
        "best_prompt_ids": best_prompt_ids,
        "best_loss": best_loss,
        "initial_loss": (float(initial_loss) if initial_loss is not None else None),
        "best_step_idx": int(best_step_idx),
        "initial_free_ids": [int(x) for x in initial_free.tolist()],
        "best_free_ids": [int(x) for x in best_free.tolist()],
        "num_accepted_updates": int(accepted_updates),
        "num_control_token_changes": int(control_token_changes_total),
        "final_no_improve_steps": int(no_improve_steps),
        "success_found": success_found,
        "success_step": success_step,
        "stop_reason": stop_reason,
        "search_steps": search_steps,
        "forward_calls": forward_calls,
        "backward_calls": backward_calls,
        "generate_calls": generate_calls,
        "best_prompt_generation_ids": best_prompt_generation_ids,
        "best_prompt_generation_success": best_prompt_success,
    }


@torch.no_grad()
def greedy_exact_match(
    model,
    tokenizer,
    prompt_ids: Sequence[int],
    target_ids: Sequence[int],
    device: torch.device,
) -> Tuple[bool, List[int]]:
    if len(prompt_ids) == 0:
        raise ValueError("Prompt must have at least one token.")

    input_ids = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
    out = model.generate(
        input_ids=input_ids,
        max_new_tokens=len(target_ids),
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    gen = out[0, input_ids.shape[1]: input_ids.shape[1] + len(target_ids)].tolist()
    success = gen == list(target_ids)
    return success, gen


def greedy_exact_match_cached(
    model,
    tokenizer,
    prompt_ids: Sequence[int],
    target_ids: Sequence[int],
    device: torch.device,
    generation_success_cache: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], Tuple[bool, List[int]]],
) -> Tuple[bool, List[int], bool]:
    cache_key = (
        tuple(int(x) for x in prompt_ids),
        tuple(int(x) for x in target_ids),
    )
    cache_entry = generation_success_cache.get(cache_key)
    if cache_entry is not None:
        success, generated_ids = cache_entry
        return bool(success), [int(x) for x in generated_ids], False

    success, generated_ids = greedy_exact_match(
        model=model,
        tokenizer=tokenizer,
        prompt_ids=prompt_ids,
        target_ids=target_ids,
        device=device,
    )
    generation_success_cache[cache_key] = (bool(success), [int(x) for x in generated_ids])
    return bool(success), [int(x) for x in generated_ids], True


def anchored_start_match_metrics(
    generated_ids: Sequence[int],
    target_ids: Sequence[int],
) -> Dict[str, Any]:
    target_len = len(target_ids)
    if target_len <= 0:
        raise ValueError("target_ids must be non-empty for anchored match metrics.")

    match_tokens = 0
    for gen_tok, tgt_tok in zip(generated_ids, target_ids):
        if int(gen_tok) != int(tgt_tok):
            break
        match_tokens += 1

    match_ratio = float(match_tokens) / float(target_len)
    match_percent = 100.0 * match_ratio
    return {
        "anchored_match_tokens": int(match_tokens),
        "anchored_match_ratio": float(match_ratio),
        "anchored_match_percent": float(match_percent),
    }


def compute_prompt_recall_diagnostics(
    model,
    tokenizer,
    prompt_ids: Sequence[int],
    target_ids: Sequence[int],
    device: torch.device,
    generation_success_cache: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], Tuple[bool, List[int]]],
    include_divergence_stats: bool = True,
) -> Dict[str, Any]:
    success, generated_ids, used_generate_call = greedy_exact_match_cached(
        model=model,
        tokenizer=tokenizer,
        prompt_ids=prompt_ids,
        target_ids=target_ids,
        device=device,
        generation_success_cache=generation_success_cache,
    )
    match = anchored_start_match_metrics(generated_ids, target_ids)
    lcp_tokens, first_div_pos = lcp_and_first_divergence_position(generated_ids, target_ids)
    target_prob = None
    target_rank = None
    forward_calls_used = 0
    if include_divergence_stats:
        target_prob, target_rank, forward_calls_used = target_token_prob_and_rank_at_divergence(
            model=model,
            prompt_ids=prompt_ids,
            generated_ids=generated_ids,
            target_ids=target_ids,
            first_divergence_position=first_div_pos,
            device=device,
        )
    return {
        "success": bool(success),
        "generated_ids": [int(x) for x in generated_ids],
        "generated_hash": hash_int_ids(generated_ids),
        "suffix_tokens": int(len(target_ids)),
        "lcp_tokens": int(lcp_tokens),
        "r_i": float(match["anchored_match_ratio"]),
        "exact_i": bool(int(lcp_tokens) == int(len(target_ids))),
        "first_divergence_position": first_div_pos,
        "target_token_prob_at_first_divergence": target_prob,
        "target_token_rank_at_first_divergence": target_rank,
        "generate_calls_used": int(used_generate_call),
        "forward_calls_used": int(forward_calls_used),
    }


# ----------------------------- evaluation -----------------------------


def build_target_compute_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        compute = r.get("compute", {})
        out.append(
            {
                "pairing_key": r.get("pairing_key"),
                "dataset_name": r.get("dataset_name"),
                "dataset_config": r.get("dataset_config"),
                "source_split": r.get("source_split"),
                "duplication_count": r.get("duplication_count"),
                "source_id": r.get("source_id"),
                "target_len": r.get("target_len"),
                "span_slot": r.get("span_slot"),
                "target_start_token_idx": r.get("target_start_token_idx"),
                "target_end_token_idx": r.get("target_end_token_idx"),
                "ourdef_any_success": r.get("ourdef_any_success"),
                "ourdef_success_count": r.get("ourdef_success_count"),
                "ourdef_distinct_success_count": r.get("ourdef_distinct_success_count"),
                "ourdef_objective_mode": r.get("ourdef_objective_mode"),
                "ourdef_objective_any_success": r.get("ourdef_objective_any_success"),
                "ourdef_objective_success_count": r.get("ourdef_objective_success_count"),
                "ourdef_objective_distinct_success_count": r.get("ourdef_objective_distinct_success_count"),
                "ourdef_dual_any_success_paraphrased": r.get("ourdef_dual_any_success_paraphrased"),
                "ourdef_dual_any_success_original": r.get("ourdef_dual_any_success_original"),
                "ourdef_dual_any_success_union": r.get("ourdef_dual_any_success_union"),
                "ourdef_dual_success_count_paraphrased": r.get("ourdef_dual_success_count_paraphrased"),
                "ourdef_dual_success_count_original": r.get("ourdef_dual_success_count_original"),
                "ourdef_dual_success_count_both": r.get("ourdef_dual_success_count_both"),
                "ourdef_dual_success_count_union": r.get("ourdef_dual_success_count_union"),
                "ourdef_dual_mean_paraphrased_match_ratio": r.get("ourdef_dual_mean_paraphrased_match_ratio"),
                "ourdef_dual_mean_original_match_ratio": r.get("ourdef_dual_mean_original_match_ratio"),
                "ourdef_best_loss": r.get("ourdef_best_loss"),
                "ourdef_most_true_restart_idx": r.get("ourdef_most_true_restart_idx"),
                "ourdef_most_true_match_tokens": r.get("ourdef_most_true_match_tokens"),
                "ourdef_most_true_match_percent": r.get("ourdef_most_true_match_percent"),
                "ourdef_most_true_match_ratio": r.get("ourdef_most_true_match_ratio"),
                "ourdef_most_true_best_loss": r.get("ourdef_most_true_best_loss"),
                "ourdef_objective_most_true_restart_idx": r.get("ourdef_objective_most_true_restart_idx"),
                "ourdef_objective_most_true_match_tokens": r.get("ourdef_objective_most_true_match_tokens"),
                "ourdef_objective_most_true_match_percent": r.get("ourdef_objective_most_true_match_percent"),
                "ourdef_objective_most_true_match_ratio": r.get("ourdef_objective_most_true_match_ratio"),
                "ourdef_objective_most_true_best_loss": r.get("ourdef_objective_most_true_best_loss"),
                "audit_restarts_executed": r.get("audit_restarts_executed"),
                "audit_discrete_optimizer": r.get("audit_discrete_optimizer"),
                "audit_gcg_no_improve_patience": r.get("audit_gcg_no_improve_patience"),
                "compute.time_sec": compute.get("time_sec"),
                "compute.search_steps": compute.get("search_steps"),
                "compute.forward_calls": compute.get("forward_calls"),
                "compute.backward_calls": compute.get("backward_calls"),
                "compute.generate_calls": compute.get("generate_calls"),
                "compute.model_calls": compute.get("model_calls"),
            }
        )
    return out


def summarize_results(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}

    n = len(rows)
    by_dup_count = Counter(int(r["duplication_count"]) for r in rows)
    by_dup_any_success = defaultdict(int)
    by_dup_objective_any_success = defaultdict(int)
    for r in rows:
        by_dup_any_success[int(r["duplication_count"])] += int(bool(r["ourdef_any_success"]))
        by_dup_objective_any_success[int(r["duplication_count"])] += int(
            bool(r.get("ourdef_objective_any_success"))
        )

    any_success_count = sum(int(bool(r["ourdef_any_success"])) for r in rows)
    success_count_total = sum(int(r["ourdef_success_count"]) for r in rows)
    distinct_success_count_total = sum(int(r["ourdef_distinct_success_count"]) for r in rows)
    objective_any_success_count = sum(int(bool(r.get("ourdef_objective_any_success"))) for r in rows)
    objective_success_count_total = sum(int(r.get("ourdef_objective_success_count", 0)) for r in rows)
    objective_distinct_success_count_total = sum(
        int(r.get("ourdef_objective_distinct_success_count", 0)) for r in rows
    )
    dual_any_paraphrased_count = sum(int(bool(r.get("ourdef_dual_any_success_paraphrased"))) for r in rows)
    dual_any_original_count = sum(int(bool(r.get("ourdef_dual_any_success_original"))) for r in rows)
    dual_any_union_count = sum(int(bool(r.get("ourdef_dual_any_success_union"))) for r in rows)
    dual_success_paraphrased_total = sum(int(r.get("ourdef_dual_success_count_paraphrased", 0)) for r in rows)
    dual_success_original_total = sum(int(r.get("ourdef_dual_success_count_original", 0)) for r in rows)
    dual_success_both_total = sum(int(r.get("ourdef_dual_success_count_both", 0)) for r in rows)
    dual_success_union_total = sum(int(r.get("ourdef_dual_success_count_union", 0)) for r in rows)
    dual_mean_paraphrased_match_ratio = (
        sum(_to_float(r.get("ourdef_dual_mean_paraphrased_match_ratio")) for r in rows) / float(n)
    )
    dual_mean_original_match_ratio = (
        sum(_to_float(r.get("ourdef_dual_mean_original_match_ratio")) for r in rows) / float(n)
    )

    return {
        "n_rows": n,
        "ourdef_any_success_count": any_success_count,
        "ourdef_any_success_rate": any_success_count / float(n),
        "ourdef_success_count_total": success_count_total,
        "ourdef_success_count_mean": success_count_total / float(n),
        "ourdef_distinct_success_count_total": distinct_success_count_total,
        "ourdef_distinct_success_count_mean": distinct_success_count_total / float(n),
        "ourdef_objective_any_success_count": objective_any_success_count,
        "ourdef_objective_any_success_rate": objective_any_success_count / float(n),
        "ourdef_objective_success_count_total": objective_success_count_total,
        "ourdef_objective_success_count_mean": objective_success_count_total / float(n),
        "ourdef_objective_distinct_success_count_total": objective_distinct_success_count_total,
        "ourdef_objective_distinct_success_count_mean": (
            objective_distinct_success_count_total / float(n)
        ),
        "ourdef_dual_any_success_paraphrased_count": dual_any_paraphrased_count,
        "ourdef_dual_any_success_paraphrased_rate": dual_any_paraphrased_count / float(n),
        "ourdef_dual_any_success_original_count": dual_any_original_count,
        "ourdef_dual_any_success_original_rate": dual_any_original_count / float(n),
        "ourdef_dual_any_success_union_count": dual_any_union_count,
        "ourdef_dual_any_success_union_rate": dual_any_union_count / float(n),
        "ourdef_dual_success_count_paraphrased_total": dual_success_paraphrased_total,
        "ourdef_dual_success_count_original_total": dual_success_original_total,
        "ourdef_dual_success_count_both_total": dual_success_both_total,
        "ourdef_dual_success_count_union_total": dual_success_union_total,
        "ourdef_dual_mean_paraphrased_match_ratio": dual_mean_paraphrased_match_ratio,
        "ourdef_dual_mean_original_match_ratio": dual_mean_original_match_ratio,
        "by_dup_count": dict(sorted(by_dup_count.items())),
        "by_dup_any_success": dict(sorted(by_dup_any_success.items())),
        "by_dup_objective_any_success": dict(sorted(by_dup_objective_any_success.items())),
    }


def summarize_compute(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}

    n = float(len(rows))

    def _sum(field: str) -> float:
        total = 0.0
        for r in rows:
            total += float(r["compute"][field])
        return total

    summary = {
        "n_targets": int(n),
        "time_sec_total": _sum("time_sec"),
        "search_steps_total": int(_sum("search_steps")),
        "forward_calls_total": int(_sum("forward_calls")),
        "backward_calls_total": int(_sum("backward_calls")),
        "generate_calls_total": int(_sum("generate_calls")),
        "model_calls_total": int(_sum("model_calls")),
    }
    summary["time_sec_mean"] = summary["time_sec_total"] / n
    summary["model_calls_mean"] = summary["model_calls_total"] / n
    return summary


# ----------------------------- CLI -----------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sampled_targets_csv",
        type=str,
        default="outputs/hubble_chat_full_1b100b_perturbed/sampled_targets.csv",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="allegrolab/hubble-1b-100b_toks-perturbed-hf",
    )

    parser.add_argument("--output_path", type=str, default="outputs/ourdef_memorization/results.csv")
    parser.add_argument("--target_compute_output_path", type=str, default=None)
    parser.add_argument("--manifest_output_path", type=str, default=None)
    parser.add_argument("--live_output_path", type=str, default=None)
    parser.add_argument("--live_target_compute_output_path", type=str, default=None)
    parser.add_argument("--live_restart_dual_output_path", type=str, default=None)
    parser.add_argument("--checkpoint_metrics_output_path", type=str, default=None)
    parser.add_argument("--loss_trajectory_output_path", type=str, default=None)
    parser.add_argument("--path_diag_output_path", type=str, default=None)

    parser.add_argument("--dup_caps", type=str, default=None)

    parser.add_argument("--num_restarts", type=int, default=5)
    parser.add_argument("--steps_per_restart", type=int, default=250)
    parser.add_argument("--adv_prefix_len", type=int, default=10)
    parser.add_argument(
        "--discrete_optimizer",
        choices=["gcg", "random_search"],
        default="gcg",
        help="Prompt optimizer used in each restart.",
    )
    parser.add_argument(
        "--scaffold_fraction",
        type=float,
        default=0.0,
        help="Requested scaffold fraction in [0, 1]. 0 uses no scaffold.",
    )
    parser.add_argument(
        "--optimize_mode",
        choices=["full_target", "continuation"],
        default="full_target",
        help="Optimization objective: full target or only continuation after scaffold.",
    )
    parser.add_argument("--gcg_no_improve_patience", type=int, default=100)
    parser.add_argument("--gcg_no_improve_patience_final", type=int, default=None)
    parser.add_argument("--candidate_topk", type=int, default=64)
    parser.add_argument("--candidate_topk_final", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--mini_batch_size", type=int, default=64)
    parser.add_argument("--gcg_position_sampling", choices=["uniform", "grad"], default="grad")
    parser.add_argument(
        "--gcg_num_mutations_per_candidate",
        type=int,
        default=1,
        help="How many token positions to mutate per candidate proposal.",
    )
    parser.add_argument(
        "--gcg_random_candidate_frac",
        type=float,
        default=0.0,
        help="Fraction of candidates that use random token mutations for exploration.",
    )
    parser.add_argument(
        "--sa_accept_worse",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable simulated-annealing style acceptance for uphill (worse-loss) proposals.",
    )
    parser.add_argument("--sa_temp_init", type=float, default=0.05)
    parser.add_argument("--sa_temp_final", type=float, default=0.005)
    parser.add_argument(
        "--dedupe_candidates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Deduplicate mutated candidates before candidate-loss evaluation.",
    )
    parser.add_argument(
        "--adaptive_schedule",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Linearly decay top-k and no-improve patience across GCG steps.",
    )
    parser.add_argument("--log_gcg_iterations", action="store_true")
    parser.add_argument("--gcg_log_every", type=int, default=25)
    parser.add_argument(
        "--checkpoint_steps",
        type=str,
        default="128,256,512",
        help="Comma-separated optimization steps for checkpoint recall diagnostics.",
    )
    parser.add_argument(
        "--trajectory_log_every",
        type=int,
        default=32,
        help="Log lightweight loss trajectory every N steps (0 disables).",
    )
    parser.add_argument(
        "--periodic_decode_every",
        type=int,
        default=0,
        help=(
            "If >0, run periodic greedy decode every N steps for threshold-crossing diagnostics. "
            "If 0, threshold crossings are inferred from checkpoint steps only."
        ),
    )
    parser.add_argument(
        "--path_id_offset",
        type=int,
        default=0,
        help="Offset added to restart index to form reported path_id (e.g., 5 -> paths 6..10).",
    )

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing live CSV files (uses row_uid when present; otherwise pairing_key with occurrence index).",
    )
    parser.add_argument(
        "--continue_on_restart_error",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Continue to the next restart when one restart fails; still logs diagnostics rows.",
    )

    return parser.parse_args()


# ----------------------------- main -----------------------------


def main() -> None:
    args = parse_args()
    run_started = time.time()

    if args.num_restarts <= 0:
        raise ValueError("--num_restarts must be positive.")
    if args.steps_per_restart <= 0:
        raise ValueError("--steps_per_restart must be positive.")
    if args.adv_prefix_len <= 0:
        raise ValueError("--adv_prefix_len must be positive.")
    if args.scaffold_fraction < 0.0 or args.scaffold_fraction > 1.0:
        raise ValueError("--scaffold_fraction must be in [0, 1].")
    if args.candidate_topk <= 0:
        raise ValueError("--candidate_topk must be positive.")
    if args.candidate_topk_final is not None and args.candidate_topk_final <= 0:
        raise ValueError("--candidate_topk_final must be positive when provided.")
    if args.batch_size <= 0 or args.mini_batch_size <= 0:
        raise ValueError("--batch_size and --mini_batch_size must be positive.")
    if args.gcg_num_mutations_per_candidate <= 0:
        raise ValueError("--gcg_num_mutations_per_candidate must be positive.")
    if args.gcg_random_candidate_frac < 0.0 or args.gcg_random_candidate_frac >= 1.0:
        raise ValueError("--gcg_random_candidate_frac must be in [0.0, 1.0).")
    if args.gcg_no_improve_patience <= 0:
        raise ValueError("--gcg_no_improve_patience must be positive.")
    if args.gcg_no_improve_patience_final is not None and args.gcg_no_improve_patience_final <= 0:
        raise ValueError("--gcg_no_improve_patience_final must be positive when provided.")
    if args.sa_temp_init <= 0 or args.sa_temp_final <= 0:
        raise ValueError("--sa_temp_init and --sa_temp_final must be positive.")
    if args.gcg_log_every <= 0:
        raise ValueError("--gcg_log_every must be positive.")
    if args.trajectory_log_every < 0:
        raise ValueError("--trajectory_log_every must be >= 0.")
    if args.periodic_decode_every < 0:
        raise ValueError("--periodic_decode_every must be >= 0.")
    if args.path_id_offset < 0:
        raise ValueError("--path_id_offset must be >= 0.")
    checkpoint_steps = parse_positive_int_csv(args.checkpoint_steps, "--checkpoint_steps")
    if len(checkpoint_steps) == 0:
        raise ValueError("--checkpoint_steps must include at least one positive step.")
    checkpoint_steps_set = set(checkpoint_steps)

    set_seed(args.seed)
    device = choose_device(args.device)
    torch_dtype = resolve_torch_dtype(args.dtype)
    load_dtype = torch_dtype if device.type == "cuda" else None

    print(
        f"[setup] model={args.model_name_or_path} device={device.type} "
        f"dtype={args.dtype} model_load_dtype={str(load_dtype) if load_dtype is not None else 'default'} "
        f"discrete_optimizer={args.discrete_optimizer}",
        flush=True,
    )
    print(
        f"[setup] gcg_position_sampling={args.gcg_position_sampling} "
        f"gcg_num_mutations_per_candidate={args.gcg_num_mutations_per_candidate} "
        f"gcg_random_candidate_frac={args.gcg_random_candidate_frac} "
        f"dedupe_candidates={int(args.dedupe_candidates)} "
        f"adaptive_schedule={int(args.adaptive_schedule)} "
        f"sa_accept_worse={int(args.sa_accept_worse)} "
        f"sa_temp_init={args.sa_temp_init} sa_temp_final={args.sa_temp_final} "
        f"optimize_mode={args.optimize_mode} "
        f"scaffold_fraction={args.scaffold_fraction} "
        f"candidate_topk={args.candidate_topk} candidate_topk_final={args.candidate_topk_final} "
        f"patience={args.gcg_no_improve_patience} patience_final={args.gcg_no_improve_patience_final}",
        flush=True,
    )
    print(
        f"[setup] checkpoint_steps={checkpoint_steps} trajectory_log_every={args.trajectory_log_every} "
        f"periodic_decode_every={args.periodic_decode_every} path_id_offset={args.path_id_offset}",
        flush=True,
    )
    if args.discrete_optimizer == "random_search":
        print(
            "[setup] random_search ignores gradient-only knobs "
            "(gcg_position_sampling, candidate_topk, candidate_topk_final, adaptive_schedule, "
            "sa_accept_worse, sa_temp_init, sa_temp_final, gcg_random_candidate_frac).",
            flush=True,
        )

    sampled_path = Path(args.sampled_targets_csv)
    if not sampled_path.exists():
        raise FileNotFoundError(f"Sampled targets file not found: {sampled_path}")

    with sampled_path.open("r", encoding="utf-8", newline="") as f:
        input_rows = list(csv.DictReader(f))

    if not input_rows:
        raise ValueError("No rows found in sampled_targets_csv.")

    if args.dup_caps is not None and str(args.dup_caps).strip() != "":
        dup_caps = parse_dup_caps(args.dup_caps)
        selected_rows, selection_diag = select_targets_by_dup(input_rows, dup_caps=dup_caps, seed=args.seed)
    else:
        selected_rows = sorted(
            list(input_rows),
            key=lambda x: (
                -int(x.get("duplication_count", 0)),
                str(x.get("dataset_name", "")),
                str(x.get("dataset_config", "")),
                str(x.get("source_split", "")),
                str(x.get("source_id", "")),
                int(x.get("target_len", 0)),
                str(x.get("span_slot", "")),
                int(x.get("target_start_token_idx", 0)),
                int(x.get("target_end_token_idx", 0)),
                str(x.get("pairing_key", "")),
            ),
        )
        selection_diag = [
            {
                "mode": "all_rows",
                "selected_total": len(selected_rows),
            }
        ]
    if not selected_rows:
        raise ValueError("No targets selected after applying dup caps.")

    print(
        f"[setup] selected_targets={len(selected_rows)} "
        f"dup_distribution={dict(sorted(Counter(int(r['duplication_count']) for r in selected_rows).items()))}",
        flush=True,
    )

    output_path = Path(args.output_path)
    if output_path.suffix == "":
        output_path = output_path.with_suffix(".csv")
    target_compute_output_path = (
        Path(args.target_compute_output_path)
        if args.target_compute_output_path is not None
        else output_path.with_name("target_compute.csv")
    )
    live_output_path = (
        Path(args.live_output_path)
        if args.live_output_path is not None
        else output_path.with_name(output_path.stem + "_live.csv")
    )
    live_target_compute_output_path = (
        Path(args.live_target_compute_output_path)
        if args.live_target_compute_output_path is not None
        else target_compute_output_path.with_name(target_compute_output_path.stem + "_live.csv")
    )
    live_restart_dual_output_path = (
        Path(args.live_restart_dual_output_path)
        if args.live_restart_dual_output_path is not None
        else live_output_path.with_name(live_output_path.stem + "_restart_dual.csv")
    )
    checkpoint_metrics_output_path = (
        Path(args.checkpoint_metrics_output_path)
        if args.checkpoint_metrics_output_path is not None
        else live_output_path.with_name(live_output_path.stem + "_checkpoint_metrics.csv")
    )
    loss_trajectory_output_path = (
        Path(args.loss_trajectory_output_path)
        if args.loss_trajectory_output_path is not None
        else live_output_path.with_name(live_output_path.stem + "_loss_trajectory.csv")
    )
    path_diag_output_path = (
        Path(args.path_diag_output_path)
        if args.path_diag_output_path is not None
        else live_output_path.with_name(live_output_path.stem + "_path_diag.csv")
    )
    manifest_output_path = (
        Path(args.manifest_output_path)
        if args.manifest_output_path is not None
        else output_path.with_name("run_manifest.json")
    )

    os.makedirs(str(live_output_path.parent), exist_ok=True)
    os.makedirs(str(live_target_compute_output_path.parent), exist_ok=True)
    os.makedirs(str(live_restart_dual_output_path.parent), exist_ok=True)
    os.makedirs(str(checkpoint_metrics_output_path.parent), exist_ok=True)
    os.makedirs(str(loss_trajectory_output_path.parent), exist_ok=True)
    os.makedirs(str(path_diag_output_path.parent), exist_ok=True)
    live_output_state: Dict[str, Any] = {"fieldnames": None}
    live_compute_state: Dict[str, Any] = {"fieldnames": None}
    live_restart_dual_state: Dict[str, Any] = {"fieldnames": None}
    checkpoint_metrics_state: Dict[str, Any] = {"fieldnames": None}
    loss_trajectory_state: Dict[str, Any] = {"fieldnames": None}
    path_diag_state: Dict[str, Any] = {"fieldnames": None}
    restart_dual_logged_keys: set[str] = set()
    checkpoint_logged_keys: set[str] = set()
    trajectory_logged_keys: set[str] = set()
    path_diag_logged_keys: set[str] = set()

    selected_resume_keys = build_resume_keys(selected_rows)
    selected_with_keys = list(zip(range(1, len(selected_rows) + 1), selected_rows, selected_resume_keys))

    existing_live_rows: List[Dict[str, Any]] = []
    done_keys: set[str] = set()
    if args.resume:
        live_exists = live_output_path.exists()
        live_compute_exists = live_target_compute_output_path.exists()
        if live_exists != live_compute_exists:
            raise ValueError(
                "--resume requires both live files to exist together: "
                f"{live_output_path} and {live_target_compute_output_path}"
            )
        if live_exists:
            existing_live_rows = read_csv_rows(live_output_path)
            done_keys = set(build_resume_keys(existing_live_rows))
            live_output_state["fieldnames"] = read_csv_header(live_output_path)
            live_compute_state["fieldnames"] = read_csv_header(live_target_compute_output_path)
            if live_restart_dual_output_path.exists():
                live_restart_dual_state["fieldnames"] = read_csv_header(live_restart_dual_output_path)
                for rr in read_csv_rows(live_restart_dual_output_path):
                    rr_resume_key = str(rr.get("resume_key", "")).strip()
                    rr_restart_idx = str(rr.get("restart_idx", "")).strip()
                    if rr_resume_key != "" and rr_restart_idx != "":
                        rr_step_idx = str(rr.get("best_step_idx", rr.get("step_idx", ""))).strip()
                        rr_event_ordinal = str(rr.get("event_ordinal", "")).strip()
                        if rr_step_idx != "" and rr_event_ordinal != "":
                            restart_dual_logged_keys.add(
                                f"{rr_resume_key}::restart{rr_restart_idx}::step{rr_step_idx}::ord{rr_event_ordinal}"
                            )
                        elif rr_step_idx != "":
                            restart_dual_logged_keys.add(
                                f"{rr_resume_key}::restart{rr_restart_idx}::step{rr_step_idx}"
                            )
                        else:
                            restart_dual_logged_keys.add(f"{rr_resume_key}::restart{rr_restart_idx}")
            if checkpoint_metrics_output_path.exists():
                checkpoint_metrics_state["fieldnames"] = read_csv_header(checkpoint_metrics_output_path)
                for cp in read_csv_rows(checkpoint_metrics_output_path):
                    cp_resume_key = str(cp.get("resume_key", "")).strip()
                    cp_restart_idx = str(cp.get("restart_idx", "")).strip()
                    cp_path_id = str(cp.get("path_id", "")).strip()
                    cp_step = str(cp.get("checkpoint_step", "")).strip()
                    cp_status = str(cp.get("status", "")).strip()
                    if cp_resume_key and cp_restart_idx and cp_path_id and cp_step and cp_status:
                        checkpoint_logged_keys.add(
                            f"{cp_resume_key}::restart{cp_restart_idx}::path{cp_path_id}::step{cp_step}::status{cp_status}"
                        )
            if loss_trajectory_output_path.exists():
                loss_trajectory_state["fieldnames"] = read_csv_header(loss_trajectory_output_path)
                for lt in read_csv_rows(loss_trajectory_output_path):
                    lt_resume_key = str(lt.get("resume_key", "")).strip()
                    lt_restart_idx = str(lt.get("restart_idx", "")).strip()
                    lt_path_id = str(lt.get("path_id", "")).strip()
                    lt_step = str(lt.get("step", "")).strip()
                    if lt_resume_key and lt_restart_idx and lt_path_id and lt_step:
                        trajectory_logged_keys.add(
                            f"{lt_resume_key}::restart{lt_restart_idx}::path{lt_path_id}::step{lt_step}"
                        )
            if path_diag_output_path.exists():
                path_diag_state["fieldnames"] = read_csv_header(path_diag_output_path)
                for pd in read_csv_rows(path_diag_output_path):
                    pd_resume_key = str(pd.get("resume_key", "")).strip()
                    pd_restart_idx = str(pd.get("restart_idx", "")).strip()
                    pd_path_id = str(pd.get("path_id", "")).strip()
                    pd_status = str(pd.get("status", "")).strip()
                    pd_step = str(pd.get("checkpoint_step", "")).strip()
                    if pd_resume_key and pd_restart_idx and pd_path_id and pd_status and pd_step:
                        path_diag_logged_keys.add(
                            f"{pd_resume_key}::restart{pd_restart_idx}::path{pd_path_id}::step{pd_step}::status{pd_status}"
                        )

            unknown_done = done_keys - set(selected_resume_keys)
            if unknown_done:
                raise ValueError(
                    "Resume checkpoint contains processed rows that are not in current selected target set. "
                    "Use matching --sampled_targets_csv or run without --resume."
                )
    else:
        if live_output_path.exists():
            live_output_path.unlink()
        if live_target_compute_output_path.exists():
            live_target_compute_output_path.unlink()
        if live_restart_dual_output_path.exists():
            live_restart_dual_output_path.unlink()
        if checkpoint_metrics_output_path.exists():
            checkpoint_metrics_output_path.unlink()
        if loss_trajectory_output_path.exists():
            loss_trajectory_output_path.unlink()
        if path_diag_output_path.exists():
            path_diag_output_path.unlink()

    done_before_run = len(done_keys)
    pending_items = [(i, r, k) for (i, r, k) in selected_with_keys if k not in done_keys]
    print(
        f"[setup] pending_targets={len(pending_items)} resume_done={done_before_run}",
        flush=True,
    )
    print(
        f"[setup] restart_dual_live_csv={live_restart_dual_output_path}",
        flush=True,
    )
    print(
        f"[setup] checkpoint_metrics_csv={checkpoint_metrics_output_path} "
        f"loss_trajectory_csv={loss_trajectory_output_path} "
        f"path_diag_csv={path_diag_output_path}",
        flush=True,
    )

    tokenizer = None
    model = None
    embedding_matrix = None
    if pending_items:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=load_dtype,
            low_cpu_mem_usage=True,
        ).to(device)
        model.eval()
        embedding_matrix = model.get_input_embeddings().weight

    rows: List[Dict[str, Any]] = []
    overall_started = time.time()

    for run_idx, (global_idx, base, resume_key) in enumerate(pending_items, 1):
        pairing_key = base["pairing_key"]
        target_ids = parse_token_ids(base["target_token_ids"])
        if len(target_ids) < 2:
            raise ValueError(f"Target too short for robust audit (len={len(target_ids)}), pairing_key={pairing_key}")

        if args.scaffold_fraction <= 0.0:
            scaffold_len = 0
        else:
            scaffold_len = int(math.floor(args.scaffold_fraction * len(target_ids)))
            scaffold_len = max(1, min(len(target_ids) - 1, scaffold_len))
        effective_scaffold_fraction = float(scaffold_len / len(target_ids))
        scaffold_ids: List[int] = []
        if scaffold_len > 0:
            scaffold_ids = target_ids[:scaffold_len]
        if args.optimize_mode == "continuation":
            objective_target_ids = target_ids[scaffold_len:]
        else:
            objective_target_ids = list(target_ids)
        if len(objective_target_ids) <= 0:
            raise ValueError(
                f"Objective target is empty under optimize_mode={args.optimize_mode}, pairing_key={pairing_key}"
            )

        original_target_ids = parse_optional_token_ids(base.get("original_target_token_ids"))
        if original_target_ids is None:
            original_target_ids = list(target_ids)
        if len(original_target_ids) <= 0:
            original_target_ids = list(target_ids)
        original_scaffold_len = scaffold_len
        original_continuation_fallback_full_target = False
        if args.optimize_mode == "continuation":
            max_valid_scaffold = max(0, len(original_target_ids) - 1)
            original_scaffold_len = min(scaffold_len, max_valid_scaffold)
            if original_scaffold_len < 0:
                original_scaffold_len = 0
            original_objective_target_ids = original_target_ids[original_scaffold_len:]
            if len(original_objective_target_ids) <= 0:
                original_objective_target_ids = list(original_target_ids)
                original_continuation_fallback_full_target = True
        else:
            original_scaffold_len = 0
            original_objective_target_ids = list(original_target_ids)
        if len(original_objective_target_ids) <= 0:
            raise ValueError(
                f"Original objective target is empty, pairing_key={pairing_key}"
            )

        success_count = 0
        distinct_keys = set()
        success_prefixes_raw: List[Dict[str, Any]] = []
        success_prefixes_distinct: List[Dict[str, Any]] = []
        objective_success_count = 0
        objective_distinct_keys = set()
        objective_success_prefixes_raw: List[Dict[str, Any]] = []
        objective_success_prefixes_distinct: List[Dict[str, Any]] = []
        dual_success_original_count = 0
        dual_success_paraphrased_count = 0
        dual_success_both_count = 0
        dual_restart_success_type_trace: List[str] = []
        dual_restart_paraphrased_match_ratio_trace: List[float] = []
        dual_restart_original_match_ratio_trace: List[float] = []
        restart_generation_matches: List[Dict[str, Any]] = []
        best_loss = float("inf")
        restarts_executed = 0
        restart_stop_reasons: List[str] = []

        compute = {
            "time_sec": 0.0,
            "search_steps": 0,
            "forward_calls": 0,
            "backward_calls": 0,
            "generate_calls": 0,
            "model_calls": 0,
        }
        generation_success_cache: Dict[
            Tuple[Tuple[int, ...], Tuple[int, ...]],
            Tuple[bool, List[int]],
        ] = {}

        target_started = time.perf_counter()

        for restart_idx in range(args.num_restarts):
            restarts_executed += 1
            logical_restart_idx = int(args.path_id_offset + restart_idx)
            restart_seed = (args.seed * 1_000_003 + global_idx * 10_007 + logical_restart_idx) % (2 ** 31 - 1)
            path_id = int(logical_restart_idx + 1)
            set_seed(restart_seed)
            restart_started = time.perf_counter()
            if device.type == "cuda" and torch.cuda.is_available():
                try:
                    torch.cuda.reset_peak_memory_stats(device)
                except Exception:
                    pass
            restart_dual_best_events: List[Dict[str, Any]] = []
            checkpoint_steps_seen: set[int] = set()
            extra_generate_calls = 0
            extra_forward_calls = 0
            first_threshold_step: Dict[float, Optional[int]] = {
                0.25: None,
                0.50: None,
                0.75: None,
                1.00: None,
            }

            def _on_new_best(event: Dict[str, Any]) -> None:
                generated_ids_evt = [int(x) for x in event.get("generated_ids", [])]
                para_evt = anchored_start_match_metrics(generated_ids_evt, objective_target_ids)
                orig_evt = anchored_start_match_metrics(generated_ids_evt, original_objective_target_ids)
                exact_para_evt = int(para_evt["anchored_match_tokens"]) == len(objective_target_ids)
                exact_orig_evt = int(orig_evt["anchored_match_tokens"]) == len(original_objective_target_ids)
                if exact_para_evt and exact_orig_evt:
                    evt_type = "both"
                elif exact_para_evt:
                    evt_type = "paraphrased"
                elif exact_orig_evt:
                    evt_type = "original"
                else:
                    evt_type = "none"

                event_ordinal = len(restart_dual_best_events) + 1
                event_row = {
                    "event_ordinal": event_ordinal,
                    "best_step_idx": int(event.get("step_idx", 0)),
                    "best_loss": float(event.get("best_loss", float("inf"))),
                    "source": str(event.get("source", "new_best")),
                    "prompt_ids": [int(x) for x in event.get("prompt_ids", [])],
                    "generated_ids": generated_ids_evt,
                    "generated_len": len(generated_ids_evt),
                    "dual_paraphrased_cont_match_tokens": int(para_evt["anchored_match_tokens"]),
                    "dual_paraphrased_cont_match_ratio": float(para_evt["anchored_match_ratio"]),
                    "dual_paraphrased_cont_match_percent": float(para_evt["anchored_match_percent"]),
                    "dual_original_cont_match_tokens": int(orig_evt["anchored_match_tokens"]),
                    "dual_original_cont_match_ratio": float(orig_evt["anchored_match_ratio"]),
                    "dual_original_cont_match_percent": float(orig_evt["anchored_match_percent"]),
                    "dual_exact_paraphrased_continuation": bool(exact_para_evt),
                    "dual_exact_original_continuation": bool(exact_orig_evt),
                    "dual_success_type": evt_type,
                    "objective_success_found": bool(event.get("success", False)),
                }
                restart_dual_best_events.append(event_row)

                restart_dual_row = {
                    "resume_key": resume_key,
                    "pairing_key": pairing_key,
                    "duplication_count": base.get("duplication_count"),
                    "dataset_name": base.get("dataset_name"),
                    "dataset_config": base.get("dataset_config"),
                    "source_split": base.get("source_split"),
                    "source_id": base.get("source_id"),
                    "run_idx": run_idx,
                    "global_idx": global_idx,
                    "restart_idx": restart_idx,
                    "path_id": path_id,
                    "restart_seed": restart_seed,
                    "event_ordinal": event_ordinal,
                    "best_step_idx": int(event.get("step_idx", 0)),
                    "event_source": str(event.get("source", "new_best")),
                    "audit_num_restarts": args.num_restarts,
                    "audit_steps_per_restart": args.steps_per_restart,
                    "optimize_mode": args.optimize_mode,
                    "scaffold_fraction_requested": float(args.scaffold_fraction),
                    "scaffold_fraction_effective": float(effective_scaffold_fraction),
                    "scaffold_token_len_paraphrased": scaffold_len,
                    "scaffold_token_len_original_eval": original_scaffold_len,
                    "original_continuation_fallback_full_target": bool(original_continuation_fallback_full_target),
                    "paraphrased_target_token_len": len(target_ids),
                    "paraphrased_continuation_token_len": len(objective_target_ids),
                    "original_target_token_len": len(original_target_ids),
                    "original_continuation_token_len": len(original_objective_target_ids),
                    "objective_generated_token_len": len(generated_ids_evt),
                    "best_loss": float(event.get("best_loss", float("inf"))),
                    "objective_success_found": bool(event.get("success", False)),
                    "dual_paraphrased_cont_match_tokens": int(para_evt["anchored_match_tokens"]),
                    "dual_paraphrased_cont_match_ratio": float(para_evt["anchored_match_ratio"]),
                    "dual_paraphrased_cont_match_percent": float(para_evt["anchored_match_percent"]),
                    "dual_original_cont_match_tokens": int(orig_evt["anchored_match_tokens"]),
                    "dual_original_cont_match_ratio": float(orig_evt["anchored_match_ratio"]),
                    "dual_original_cont_match_percent": float(orig_evt["anchored_match_percent"]),
                    "dual_exact_paraphrased_continuation": bool(exact_para_evt),
                    "dual_exact_original_continuation": bool(exact_orig_evt),
                    "dual_success_type": evt_type,
                }
                restart_log_key = (
                    f"{resume_key}::restart{restart_idx}::path{path_id}::step{int(event.get('step_idx', 0))}::ord{event_ordinal}"
                )
                if restart_log_key not in restart_dual_logged_keys:
                    append_table_row(
                        path=str(live_restart_dual_output_path),
                        row=restart_dual_row,
                        fieldnames_state=live_restart_dual_state,
                        delimiter=",",
                    )
                    restart_dual_logged_keys.add(restart_log_key)

                print(
                    f"[dual_eval_best] target={pairing_key} restart={restart_idx + 1}/{args.num_restarts} path_id={path_id} "
                    f"event={event_ordinal} step={int(event.get('step_idx', 0))} "
                    f"paraphrased_ratio={float(para_evt['anchored_match_ratio']):.4f} "
                    f"original_ratio={float(orig_evt['anchored_match_ratio']):.4f} "
                    f"exact_paraphrased={int(exact_para_evt)} exact_original={int(exact_orig_evt)} "
                    f"type={evt_type}",
                    flush=True,
                )

            def _on_step_end(event: Dict[str, Any]) -> None:
                nonlocal extra_generate_calls, extra_forward_calls

                step_idx_evt = int(event.get("step_idx", 0))
                if step_idx_evt <= 0:
                    return

                if args.trajectory_log_every > 0 and step_idx_evt % args.trajectory_log_every == 0:
                    traj_row = {
                        "resume_key": resume_key,
                        "pairing_key": pairing_key,
                        "target_id": str(base.get("row_uid", pairing_key)),
                        "duplication": int(base.get("duplication_count", 0)),
                        "path_id": path_id,
                        "restart_idx": restart_idx,
                        "seed": restart_seed,
                        "step": step_idx_evt,
                        "current_loss": float(event.get("current_loss", float("inf"))),
                        "best_loss_so_far": float(event.get("best_loss", float("inf"))),
                    }
                    traj_key = f"{resume_key}::restart{restart_idx}::path{path_id}::step{step_idx_evt}"
                    if traj_key not in trajectory_logged_keys:
                        append_table_row(
                            path=str(loss_trajectory_output_path),
                            row=traj_row,
                            fieldnames_state=loss_trajectory_state,
                            delimiter=",",
                        )
                        trajectory_logged_keys.add(traj_key)

                decode_for_checkpoint = (
                    step_idx_evt in checkpoint_steps_set and step_idx_evt not in checkpoint_steps_seen
                )
                decode_for_periodic = (
                    args.periodic_decode_every > 0 and step_idx_evt % args.periodic_decode_every == 0
                )
                if not decode_for_checkpoint and not decode_for_periodic:
                    return

                current_prompt_ids_evt = [int(x) for x in event.get("current_prompt_ids", [])]
                best_prompt_ids_evt = [int(x) for x in event.get("best_prompt_ids", [])]
                current_diag = compute_prompt_recall_diagnostics(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_ids=current_prompt_ids_evt,
                    target_ids=objective_target_ids,
                    device=device,
                    generation_success_cache=generation_success_cache,
                    include_divergence_stats=True,
                )
                extra_generate_calls += int(current_diag.get("generate_calls_used", 0))
                extra_forward_calls += int(current_diag.get("forward_calls_used", 0))
                r_i_current = float(current_diag["r_i"])
                for tau in (0.25, 0.50, 0.75, 1.00):
                    if first_threshold_step[tau] is None and r_i_current >= (tau - 1e-9):
                        first_threshold_step[tau] = int(step_idx_evt)

                if not decode_for_checkpoint:
                    return

                best_diag = compute_prompt_recall_diagnostics(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_ids=best_prompt_ids_evt,
                    target_ids=objective_target_ids,
                    device=device,
                    generation_success_cache=generation_success_cache,
                    include_divergence_stats=False,
                )
                extra_generate_calls += int(best_diag.get("generate_calls_used", 0))
                extra_forward_calls += int(best_diag.get("forward_calls_used", 0))

                checkpoint_row = {
                    "resume_key": resume_key,
                    "pairing_key": pairing_key,
                    "target_id": str(base.get("row_uid", pairing_key)),
                    "source_id": str(base.get("source_id", "")),
                    "duplication": int(base.get("duplication_count", 0)),
                    "path_id": path_id,
                    "restart_idx": restart_idx,
                    "seed": restart_seed,
                    "checkpoint_step": int(step_idx_evt),
                    "status": "ok",
                    "error_message": "",
                    "wall_time_sec": float(time.perf_counter() - restart_started),
                    "gpu_name": (
                        torch.cuda.get_device_name(device)
                        if device.type == "cuda" and torch.cuda.is_available()
                        else device.type
                    ),
                    "peak_gpu_memory_mb": peak_gpu_memory_mb(device),
                    "prefix_tokens": len(current_prompt_ids_evt),
                    "suffix_tokens": len(objective_target_ids),
                    "control_length": int(args.adv_prefix_len),
                    "current_loss": float(event.get("current_loss", float("inf"))),
                    "best_loss_so_far": float(event.get("best_loss", float("inf"))),
                    "best_step_so_far": int(event.get("best_step_idx", 0)),
                    "loss_initial": event.get("loss_initial"),
                    "r_i_current": float(current_diag["r_i"]),
                    "r_i_best_so_far": float(best_diag["r_i"]),
                    "exact_current": bool(current_diag["exact_i"]),
                    "exact_best_so_far": bool(best_diag["exact_i"]),
                    "lcp_tokens_current": int(current_diag["lcp_tokens"]),
                    "lcp_tokens_best_so_far": int(best_diag["lcp_tokens"]),
                    "first_divergence_position": current_diag["first_divergence_position"],
                    "target_token_prob_at_first_divergence": current_diag[
                        "target_token_prob_at_first_divergence"
                    ],
                    "target_token_rank_at_first_divergence": current_diag[
                        "target_token_rank_at_first_divergence"
                    ],
                    "initial_control_hash": hash_int_ids(event.get("initial_free_ids", [])),
                    "current_control_hash": hash_int_ids(event.get("current_free_ids", [])),
                    "best_control_hash": hash_int_ids(event.get("best_free_ids", [])),
                    "current_generation_hash": str(current_diag["generated_hash"]),
                    "best_generation_hash": str(best_diag["generated_hash"]),
                    "num_control_token_changes": int(event.get("num_control_token_changes", 0)),
                    "num_accepted_updates": int(event.get("num_accepted_updates", 0)),
                    "early_stop_triggered": bool(event.get("early_stop_triggered", False)),
                    "patience_counter_final": int(event.get("no_improve_steps", 0)),
                }
                cp_key = (
                    f"{resume_key}::restart{restart_idx}::path{path_id}::step{step_idx_evt}::statusok"
                )
                if cp_key not in checkpoint_logged_keys:
                    append_table_row(
                        path=str(checkpoint_metrics_output_path),
                        row=checkpoint_row,
                        fieldnames_state=checkpoint_metrics_state,
                        delimiter=",",
                    )
                    checkpoint_logged_keys.add(cp_key)
                checkpoint_steps_seen.add(int(step_idx_evt))

            optimize_fn = optimize_gcg_prefix if args.discrete_optimizer == "gcg" else optimize_random_search_prefix
            restart_error: Optional[str] = None
            try:
                opt = optimize_fn(
                    model=model,
                    tokenizer=tokenizer,
                    embedding_matrix=embedding_matrix,
                    scaffold_ids=scaffold_ids,
                    target_ids=objective_target_ids,
                    adv_prefix_len=args.adv_prefix_len,
                    num_steps=args.steps_per_restart,
                    no_improve_patience=args.gcg_no_improve_patience,
                    topk=args.candidate_topk,
                    batch_size=args.batch_size,
                    mini_batch_size=args.mini_batch_size,
                    device=device,
                    generation_success_cache=generation_success_cache,
                    position_sampling=args.gcg_position_sampling,
                    num_mutations_per_candidate=args.gcg_num_mutations_per_candidate,
                    random_candidate_frac=args.gcg_random_candidate_frac,
                    dedupe_candidates=args.dedupe_candidates,
                    adaptive_schedule=args.adaptive_schedule,
                    topk_final=args.candidate_topk_final,
                    no_improve_patience_final=args.gcg_no_improve_patience_final,
                    sa_accept_worse=args.sa_accept_worse,
                    sa_temp_init=args.sa_temp_init,
                    sa_temp_final=args.sa_temp_final,
                    log_gcg_iterations=args.log_gcg_iterations,
                    gcg_log_every=args.gcg_log_every,
                    on_new_best=_on_new_best,
                    on_step_end=_on_step_end,
                )
            except Exception as exc:
                restart_error = str(exc)
                path_diag_row = {
                    "resume_key": resume_key,
                    "pairing_key": pairing_key,
                    "target_id": str(base.get("row_uid", pairing_key)),
                    "source_id": str(base.get("source_id", "")),
                    "duplication": int(base.get("duplication_count", 0)),
                    "path_id": path_id,
                    "restart_idx": restart_idx,
                    "seed": restart_seed,
                    "checkpoint_step": -1,
                    "status": "failed",
                    "error_message": restart_error,
                    "wall_time_sec": float(time.perf_counter() - restart_started),
                    "gpu_name": (
                        torch.cuda.get_device_name(device)
                        if device.type == "cuda" and torch.cuda.is_available()
                        else device.type
                    ),
                    "peak_gpu_memory_mb": peak_gpu_memory_mb(device),
                }
                pd_key = f"{resume_key}::restart{restart_idx}::path{path_id}::step-1::statusfailed"
                if pd_key not in path_diag_logged_keys:
                    append_table_row(
                        path=str(path_diag_output_path),
                        row=path_diag_row,
                        fieldnames_state=path_diag_state,
                        delimiter=",",
                    )
                    path_diag_logged_keys.add(pd_key)
                if not args.continue_on_restart_error:
                    raise
                restart_stop_reasons.append("error")
                for missing_step in checkpoint_steps:
                    if missing_step in checkpoint_steps_seen:
                        continue
                    miss_row = {
                        "resume_key": resume_key,
                        "pairing_key": pairing_key,
                        "target_id": str(base.get("row_uid", pairing_key)),
                        "source_id": str(base.get("source_id", "")),
                        "duplication": int(base.get("duplication_count", 0)),
                        "path_id": path_id,
                        "restart_idx": restart_idx,
                        "seed": restart_seed,
                        "checkpoint_step": int(missing_step),
                        "status": "failed",
                        "error_message": restart_error,
                        "wall_time_sec": float(time.perf_counter() - restart_started),
                        "gpu_name": (
                            torch.cuda.get_device_name(device)
                            if device.type == "cuda" and torch.cuda.is_available()
                            else device.type
                        ),
                        "peak_gpu_memory_mb": peak_gpu_memory_mb(device),
                    }
                    miss_key = (
                        f"{resume_key}::restart{restart_idx}::path{path_id}::step{missing_step}::statusfailed"
                    )
                    if miss_key not in checkpoint_logged_keys:
                        append_table_row(
                            path=str(checkpoint_metrics_output_path),
                            row=miss_row,
                            fieldnames_state=checkpoint_metrics_state,
                            delimiter=",",
                        )
                        checkpoint_logged_keys.add(miss_key)
                continue

            compute["search_steps"] += int(opt["search_steps"])
            compute["forward_calls"] += int(opt["forward_calls"])
            compute["backward_calls"] += int(opt["backward_calls"])
            compute["generate_calls"] += int(opt["generate_calls"])
            compute["forward_calls"] += int(extra_forward_calls)
            compute["generate_calls"] += int(extra_generate_calls)

            if float(opt["best_loss"]) < best_loss:
                best_loss = float(opt["best_loss"])

            prompt_ids = opt["best_prompt_ids"].tolist()
            objective_generated_ids = [int(x) for x in opt["best_prompt_generation_ids"]]
            objective_success = bool(opt["success_found"])
            objective_best_prompt_generation_success = bool(opt["best_prompt_generation_success"])
            full_success, generated_ids, used_generate_call = greedy_exact_match_cached(
                model=model,
                tokenizer=tokenizer,
                prompt_ids=prompt_ids,
                target_ids=target_ids,
                device=device,
                generation_success_cache=generation_success_cache,
            )
            compute["generate_calls"] += int(used_generate_call)
            legacy_success = bool(opt["success_found"]) if args.optimize_mode == "full_target" else bool(full_success)
            restart_stop_reasons.append(str(opt["stop_reason"]))
            match_metrics = anchored_start_match_metrics(generated_ids, target_ids)
            objective_match_metrics = anchored_start_match_metrics(
                objective_generated_ids,
                objective_target_ids,
            )
            original_objective_match_metrics = anchored_start_match_metrics(
                objective_generated_ids,
                original_objective_target_ids,
            )
            dual_exact_paraphrased = (
                int(objective_match_metrics["anchored_match_tokens"]) == len(objective_target_ids)
            )
            dual_exact_original = (
                int(original_objective_match_metrics["anchored_match_tokens"]) == len(original_objective_target_ids)
            )
            if dual_exact_paraphrased and dual_exact_original:
                dual_success_type = "both"
            elif dual_exact_paraphrased:
                dual_success_type = "paraphrased"
            elif dual_exact_original:
                dual_success_type = "original"
            else:
                dual_success_type = "none"
            if dual_exact_paraphrased:
                dual_success_paraphrased_count += 1
            if dual_exact_original:
                dual_success_original_count += 1
            if dual_success_type == "both":
                dual_success_both_count += 1
            dual_restart_success_type_trace.append(dual_success_type)
            dual_restart_paraphrased_match_ratio_trace.append(
                float(objective_match_metrics["anchored_match_ratio"])
            )
            dual_restart_original_match_ratio_trace.append(
                float(original_objective_match_metrics["anchored_match_ratio"])
            )

            restart_entry = {
                "restart_idx": restart_idx,
                "path_id": path_id,
                "restart_seed": restart_seed,
                "success_step": opt["success_step"],
                "stop_reason": opt["stop_reason"],
                "best_loss": float(opt["best_loss"]),
                "initial_loss": opt.get("initial_loss"),
                "best_step_idx": opt.get("best_step_idx"),
                "num_accepted_updates": opt.get("num_accepted_updates"),
                "num_control_token_changes": opt.get("num_control_token_changes"),
                "final_no_improve_steps": opt.get("final_no_improve_steps"),
                "success_found": bool(legacy_success),
                "objective_success_found": bool(objective_success),
                "objective_best_prompt_generation_success": bool(objective_best_prompt_generation_success),
                "prefix_token_ids": prompt_ids,
                "prefix_text": tokenizer.decode(prompt_ids, skip_special_tokens=True),
                "generated_token_ids": generated_ids,
                "generated_text": tokenizer.decode(generated_ids, skip_special_tokens=True),
                "generated_token_len": len(generated_ids),
                "target_token_len": len(target_ids),
                "objective_generated_token_ids": objective_generated_ids,
                "objective_generated_text": tokenizer.decode(
                    objective_generated_ids, skip_special_tokens=True
                ),
                "objective_generated_token_len": len(objective_generated_ids),
                "objective_target_token_len": len(objective_target_ids),
                **match_metrics,
                "objective_anchored_match_tokens": int(objective_match_metrics["anchored_match_tokens"]),
                "objective_anchored_match_ratio": float(objective_match_metrics["anchored_match_ratio"]),
                "objective_anchored_match_percent": float(objective_match_metrics["anchored_match_percent"]),
                "dual_paraphrased_cont_match_tokens": int(objective_match_metrics["anchored_match_tokens"]),
                "dual_paraphrased_cont_match_ratio": float(objective_match_metrics["anchored_match_ratio"]),
                "dual_paraphrased_cont_match_percent": float(objective_match_metrics["anchored_match_percent"]),
                "dual_original_cont_match_tokens": int(original_objective_match_metrics["anchored_match_tokens"]),
                "dual_original_cont_match_ratio": float(original_objective_match_metrics["anchored_match_ratio"]),
                "dual_original_cont_match_percent": float(original_objective_match_metrics["anchored_match_percent"]),
                "dual_exact_paraphrased_continuation": bool(dual_exact_paraphrased),
                "dual_exact_original_continuation": bool(dual_exact_original),
                "dual_success_type": dual_success_type,
                "dual_new_best_events": restart_dual_best_events,
            }
            restart_generation_matches.append(restart_entry)

            if legacy_success:
                success_count += 1

                entry = {
                    "restart_idx": restart_idx,
                    "path_id": path_id,
                    "restart_seed": restart_seed,
                    "success_step": opt["success_step"],
                    "stop_reason": opt["stop_reason"],
                    "best_loss": float(opt["best_loss"]),
                    "prefix_token_ids": prompt_ids,
                    "prefix_text": tokenizer.decode(prompt_ids, skip_special_tokens=True),
                    "generated_token_ids": generated_ids,
                    "generated_text": tokenizer.decode(generated_ids, skip_special_tokens=True),
                    **match_metrics,
                }
                success_prefixes_raw.append(entry)

                key = tuple(prompt_ids)
                if key not in distinct_keys:
                    distinct_keys.add(key)
                    success_prefixes_distinct.append(entry)
            if objective_success:
                objective_success_count += 1
                objective_entry = {
                    "restart_idx": restart_idx,
                    "path_id": path_id,
                    "restart_seed": restart_seed,
                    "success_step": opt["success_step"],
                    "stop_reason": opt["stop_reason"],
                    "best_loss": float(opt["best_loss"]),
                    "prefix_token_ids": prompt_ids,
                    "prefix_text": tokenizer.decode(prompt_ids, skip_special_tokens=True),
                    "generated_token_ids": objective_generated_ids,
                    "generated_text": tokenizer.decode(objective_generated_ids, skip_special_tokens=True),
                    "anchored_match_tokens": int(objective_match_metrics["anchored_match_tokens"]),
                    "anchored_match_ratio": float(objective_match_metrics["anchored_match_ratio"]),
                    "anchored_match_percent": float(objective_match_metrics["anchored_match_percent"]),
                }
                objective_success_prefixes_raw.append(objective_entry)
                objective_key = tuple(prompt_ids)
                if objective_key not in objective_distinct_keys:
                    objective_distinct_keys.add(objective_key)
                    objective_success_prefixes_distinct.append(objective_entry)

            for missing_step in checkpoint_steps:
                if missing_step in checkpoint_steps_seen:
                    continue
                miss_row = {
                    "resume_key": resume_key,
                    "pairing_key": pairing_key,
                    "target_id": str(base.get("row_uid", pairing_key)),
                    "source_id": str(base.get("source_id", "")),
                    "duplication": int(base.get("duplication_count", 0)),
                    "path_id": path_id,
                    "restart_idx": restart_idx,
                    "seed": restart_seed,
                    "checkpoint_step": int(missing_step),
                    "status": "not_reached",
                    "error_message": "",
                    "wall_time_sec": float(time.perf_counter() - restart_started),
                    "gpu_name": (
                        torch.cuda.get_device_name(device)
                        if device.type == "cuda" and torch.cuda.is_available()
                        else device.type
                    ),
                    "peak_gpu_memory_mb": peak_gpu_memory_mb(device),
                    "prefix_tokens": int(scaffold_len + args.adv_prefix_len),
                    "suffix_tokens": len(objective_target_ids),
                    "control_length": int(args.adv_prefix_len),
                    "current_loss": None,
                    "best_loss_so_far": float(opt["best_loss"]),
                    "best_step_so_far": int(opt.get("best_step_idx", 0)),
                    "loss_initial": opt.get("initial_loss"),
                    "r_i_current": None,
                    "r_i_best_so_far": None,
                    "exact_current": None,
                    "exact_best_so_far": None,
                    "lcp_tokens_current": None,
                    "lcp_tokens_best_so_far": None,
                    "first_divergence_position": None,
                    "target_token_prob_at_first_divergence": None,
                    "target_token_rank_at_first_divergence": None,
                    "initial_control_hash": hash_int_ids(opt.get("initial_free_ids", [])),
                    "current_control_hash": None,
                    "best_control_hash": hash_int_ids(opt.get("best_free_ids", [])),
                    "current_generation_hash": None,
                    "best_generation_hash": hash_int_ids(opt.get("best_prompt_generation_ids", [])),
                    "num_control_token_changes": int(opt.get("num_control_token_changes", 0)),
                    "num_accepted_updates": int(opt.get("num_accepted_updates", 0)),
                    "early_stop_triggered": bool(opt.get("stop_reason", "") != "max_steps"),
                    "patience_counter_final": int(opt.get("final_no_improve_steps", 0)),
                }
                miss_key = (
                    f"{resume_key}::restart{restart_idx}::path{path_id}::step{missing_step}::statusnot_reached"
                )
                if miss_key not in checkpoint_logged_keys:
                    append_table_row(
                        path=str(checkpoint_metrics_output_path),
                        row=miss_row,
                        fieldnames_state=checkpoint_metrics_state,
                        delimiter=",",
                    )
                    checkpoint_logged_keys.add(miss_key)

            path_diag_row = {
                "resume_key": resume_key,
                "pairing_key": pairing_key,
                "target_id": str(base.get("row_uid", pairing_key)),
                "source_id": str(base.get("source_id", "")),
                "duplication": int(base.get("duplication_count", 0)),
                "path_id": path_id,
                "restart_idx": restart_idx,
                "seed": restart_seed,
                "checkpoint_step": int(opt.get("search_steps", 0)),
                "status": "ok",
                "error_message": "",
                "wall_time_sec": float(time.perf_counter() - restart_started),
                "gpu_name": (
                    torch.cuda.get_device_name(device)
                    if device.type == "cuda" and torch.cuda.is_available()
                    else device.type
                ),
                "peak_gpu_memory_mb": peak_gpu_memory_mb(device),
                "prefix_tokens": int(scaffold_len + args.adv_prefix_len),
                "suffix_tokens": len(objective_target_ids),
                "control_length": int(args.adv_prefix_len),
                "current_loss": None,
                "best_loss_so_far": float(opt["best_loss"]),
                "best_step_so_far": int(opt.get("best_step_idx", 0)),
                "loss_initial": opt.get("initial_loss"),
                "r_i_current": None,
                "r_i_best_so_far": None,
                "exact_current": None,
                "exact_best_so_far": bool(objective_best_prompt_generation_success),
                "lcp_tokens_current": None,
                "lcp_tokens_best_so_far": int(objective_match_metrics["anchored_match_tokens"]),
                "first_divergence_position": None,
                "target_token_prob_at_first_divergence": None,
                "target_token_rank_at_first_divergence": None,
                "initial_control_hash": hash_int_ids(opt.get("initial_free_ids", [])),
                "current_control_hash": None,
                "best_control_hash": hash_int_ids(opt.get("best_free_ids", [])),
                "current_generation_hash": None,
                "best_generation_hash": hash_int_ids(objective_generated_ids),
                "num_control_token_changes": int(opt.get("num_control_token_changes", 0)),
                "num_accepted_updates": int(opt.get("num_accepted_updates", 0)),
                "early_stop_triggered": bool(opt.get("stop_reason", "") != "max_steps"),
                "patience_counter_final": int(opt.get("final_no_improve_steps", 0)),
                "first_step_T025": first_threshold_step[0.25],
                "first_step_T050": first_threshold_step[0.50],
                "first_step_T075": first_threshold_step[0.75],
                "first_step_T100": first_threshold_step[1.00],
                "crossed_T025_by_checkpoints": bool(first_threshold_step[0.25] is not None),
                "crossed_T050_by_checkpoints": bool(first_threshold_step[0.50] is not None),
                "crossed_T075_by_checkpoints": bool(first_threshold_step[0.75] is not None),
                "crossed_T100_by_checkpoints": bool(first_threshold_step[1.00] is not None),
            }
            pd_key = (
                f"{resume_key}::restart{restart_idx}::path{path_id}::step{int(opt.get('search_steps', 0))}::statusok"
            )
            if pd_key not in path_diag_logged_keys:
                append_table_row(
                    path=str(path_diag_output_path),
                    row=path_diag_row,
                    fieldnames_state=path_diag_state,
                    delimiter=",",
                )
                path_diag_logged_keys.add(pd_key)

        most_true_generation: Optional[Dict[str, Any]] = None
        objective_most_true_generation: Optional[Dict[str, Any]] = None
        if restart_generation_matches:
            def _most_true_sort_key(item: Dict[str, Any]) -> Tuple[float, float, float, int]:
                candidate_loss = float(item.get("best_loss", float("inf")))
                if not math.isfinite(candidate_loss):
                    candidate_loss = float("inf")
                return (
                    -float(item.get("anchored_match_tokens", 0)),
                    -float(item.get("anchored_match_ratio", 0.0)),
                    candidate_loss,
                    int(item.get("restart_idx", 0)),
                )

            most_true_generation = sorted(restart_generation_matches, key=_most_true_sort_key)[0]
            objective_most_true_generation = sorted(
                restart_generation_matches,
                key=lambda item: (
                    -float(item.get("objective_anchored_match_tokens", 0)),
                    -float(item.get("objective_anchored_match_ratio", 0.0)),
                    (
                        float(item.get("best_loss"))
                        if math.isfinite(float(item.get("best_loss", float("inf"))))
                        else float("inf")
                    ),
                    int(item.get("restart_idx", 0)),
                ),
            )[0]

        compute["time_sec"] = time.perf_counter() - target_started
        compute["model_calls"] = (
            compute["forward_calls"] + compute["backward_calls"] + compute["generate_calls"]
        )

        out_row: Dict[str, Any] = dict(base)
        out_row["target_token_ids"] = target_ids
        out_row.update(
            {
                "ourdef_definition": "exact_token_level_greedy_recovery_under_fixed_budget",
                "ourdef_any_success": success_count > 0,
                "ourdef_success_count": success_count,
                "ourdef_distinct_success_count": len(distinct_keys),
                "ourdef_success_prefixes": success_prefixes_distinct,
                "ourdef_success_prefixes_raw": success_prefixes_raw,
                "ourdef_objective_mode": args.optimize_mode,
                "ourdef_objective_definition": (
                    "exact_token_level_greedy_recovery_of_continuation_under_fixed_budget"
                    if args.optimize_mode == "continuation"
                    else "exact_token_level_greedy_recovery_under_fixed_budget"
                ),
                "ourdef_objective_target_token_len": len(objective_target_ids),
                "ourdef_objective_any_success": objective_success_count > 0,
                "ourdef_objective_success_count": objective_success_count,
                "ourdef_objective_distinct_success_count": len(objective_distinct_keys),
                "ourdef_objective_success_prefixes": objective_success_prefixes_distinct,
                "ourdef_objective_success_prefixes_raw": objective_success_prefixes_raw,
                "ourdef_dual_eval_mode": args.optimize_mode,
                "ourdef_dual_paraphrased_continuation_len": len(objective_target_ids),
                "ourdef_dual_original_continuation_len": len(original_objective_target_ids),
                "ourdef_dual_original_scaffold_token_len_eval": original_scaffold_len,
                "ourdef_dual_original_continuation_fallback_full_target": bool(
                    original_continuation_fallback_full_target
                ),
                "ourdef_dual_success_count_paraphrased": dual_success_paraphrased_count,
                "ourdef_dual_success_count_original": dual_success_original_count,
                "ourdef_dual_success_count_both": dual_success_both_count,
                "ourdef_dual_success_count_union": (
                    dual_success_paraphrased_count + dual_success_original_count - dual_success_both_count
                ),
                "ourdef_dual_any_success_paraphrased": dual_success_paraphrased_count > 0,
                "ourdef_dual_any_success_original": dual_success_original_count > 0,
                "ourdef_dual_any_success_union": (
                    (dual_success_paraphrased_count + dual_success_original_count - dual_success_both_count) > 0
                ),
                "ourdef_dual_restart_success_type_trace": dual_restart_success_type_trace,
                "ourdef_dual_restart_paraphrased_match_ratio_trace": dual_restart_paraphrased_match_ratio_trace,
                "ourdef_dual_restart_original_match_ratio_trace": dual_restart_original_match_ratio_trace,
                "ourdef_dual_mean_paraphrased_match_ratio": (
                    float(sum(dual_restart_paraphrased_match_ratio_trace) / len(dual_restart_paraphrased_match_ratio_trace))
                    if dual_restart_paraphrased_match_ratio_trace
                    else 0.0
                ),
                "ourdef_dual_mean_original_match_ratio": (
                    float(sum(dual_restart_original_match_ratio_trace) / len(dual_restart_original_match_ratio_trace))
                    if dual_restart_original_match_ratio_trace
                    else 0.0
                ),
                "ourdef_restart_generation_matches": restart_generation_matches,
                "ourdef_best_loss": (best_loss if math.isfinite(best_loss) else None),
                "ourdef_most_true_generation": most_true_generation,
                "ourdef_most_true_restart_idx": (
                    int(most_true_generation["restart_idx"])
                    if most_true_generation is not None
                    else None
                ),
                "ourdef_most_true_match_tokens": (
                    int(most_true_generation["anchored_match_tokens"])
                    if most_true_generation is not None
                    else None
                ),
                "ourdef_most_true_match_percent": (
                    float(most_true_generation["anchored_match_percent"])
                    if most_true_generation is not None
                    else None
                ),
                "ourdef_most_true_match_ratio": (
                    float(most_true_generation["anchored_match_ratio"])
                    if most_true_generation is not None
                    else None
                ),
                "ourdef_most_true_best_loss": (
                    float(most_true_generation["best_loss"])
                    if most_true_generation is not None and math.isfinite(float(most_true_generation["best_loss"]))
                    else None
                ),
                "ourdef_objective_most_true_generation": objective_most_true_generation,
                "ourdef_objective_most_true_restart_idx": (
                    int(objective_most_true_generation["restart_idx"])
                    if objective_most_true_generation is not None
                    else None
                ),
                "ourdef_objective_most_true_match_tokens": (
                    int(objective_most_true_generation["objective_anchored_match_tokens"])
                    if objective_most_true_generation is not None
                    else None
                ),
                "ourdef_objective_most_true_match_percent": (
                    float(objective_most_true_generation["objective_anchored_match_percent"])
                    if objective_most_true_generation is not None
                    else None
                ),
                "ourdef_objective_most_true_match_ratio": (
                    float(objective_most_true_generation["objective_anchored_match_ratio"])
                    if objective_most_true_generation is not None
                    else None
                ),
                "ourdef_objective_most_true_best_loss": (
                    float(objective_most_true_generation["best_loss"])
                    if objective_most_true_generation is not None
                    and math.isfinite(float(objective_most_true_generation["best_loss"]))
                    else None
                ),
                "ourdef_full_target_any_success": success_count > 0,
                "ourdef_full_target_success_count": success_count,
                "ourdef_full_target_distinct_success_count": len(distinct_keys),
                "audit_budget_mode": "fixed",
                "audit_discrete_optimizer": args.discrete_optimizer,
                "audit_num_restarts": args.num_restarts,
                "audit_steps_per_restart": args.steps_per_restart,
                "audit_restarts_executed": restarts_executed,
                "audit_path_id_offset": int(args.path_id_offset),
                "audit_checkpoint_steps": checkpoint_steps,
                "audit_trajectory_log_every": int(args.trajectory_log_every),
                "audit_periodic_decode_every": int(args.periodic_decode_every),
                "audit_gcg_no_improve_patience": args.gcg_no_improve_patience,
                "audit_restart_stop_reasons": restart_stop_reasons,
                "adv_prefix_len": args.adv_prefix_len,
                "scaffold_fraction": effective_scaffold_fraction,
                "scaffold_fraction_requested": float(args.scaffold_fraction),
                "scaffold_token_len": scaffold_len,
                "compute": compute,
            }
        )
        rows.append(out_row)
        out_compute_row = build_target_compute_rows([out_row])[0]
        append_table_row(
            path=str(live_output_path),
            row=out_row,
            fieldnames_state=live_output_state,
            delimiter=",",
        )
        append_table_row(
            path=str(live_target_compute_output_path),
            row=out_compute_row,
            fieldnames_state=live_compute_state,
            delimiter=",",
        )

        if args.log_every > 0 and (run_idx % args.log_every == 0 or run_idx == len(pending_items)):
            elapsed = max(1e-9, time.time() - overall_started)
            rate = run_idx / elapsed
            remaining = max(0, len(pending_items) - run_idx)
            eta = remaining / max(1e-9, rate)
            print(
                f"[progress] {run_idx}/{len(pending_items)} ({100.0 * run_idx / len(pending_items):.1f}%) "
                f"rate={rate:.2f} targets/s eta_sec={eta:.1f}",
                flush=True,
            )

    if args.resume:
        final_rows = read_csv_rows(live_output_path)
        final_target_compute_rows = read_csv_rows(live_target_compute_output_path)
        write_table(str(output_path), final_rows, delimiter=",")
        write_table(str(target_compute_output_path), final_target_compute_rows, delimiter=",")
        results_summary = summarize_results_flat(final_rows)
        compute_summary = summarize_compute_flat(final_target_compute_rows)
        output_rows_count = len(final_rows)
        output_target_compute_count = len(final_target_compute_rows)
    else:
        write_table(str(output_path), rows, delimiter=",")
        target_compute_rows = build_target_compute_rows(rows)
        write_table(str(target_compute_output_path), target_compute_rows, delimiter=",")
        results_summary = summarize_results(rows)
        compute_summary = summarize_compute(rows)
        output_rows_count = len(rows)
        output_target_compute_count = len(target_compute_rows)

    manifest = {
        "created_at_unix": time.time(),
        "wall_time_sec": time.time() - run_started,
        "args": vars(args),
        "resolved_paths": {
            "results": str(output_path),
            "target_compute_csv": str(target_compute_output_path),
            "results_live_csv": str(live_output_path),
            "target_compute_live_csv": str(live_target_compute_output_path),
            "restart_dual_live_csv": str(live_restart_dual_output_path),
            "checkpoint_metrics_csv": str(checkpoint_metrics_output_path),
            "loss_trajectory_csv": str(loss_trajectory_output_path),
            "path_diag_csv": str(path_diag_output_path),
            "manifest": str(manifest_output_path),
        },
        "selection_diagnostics": selection_diag,
        "resume": {
            "enabled": bool(args.resume),
            "done_before_run": done_before_run,
            "new_rows_added": len(rows),
            "total_rows_written": output_rows_count,
        },
        "results_summary": results_summary,
        "compute_summary": compute_summary,
    }
    write_manifest(str(manifest_output_path), manifest)

    print(
        f"[summary] n_rows={results_summary.get('n_rows', 0)} "
        f"ourdef_any_success_rate={results_summary.get('ourdef_any_success_rate', 0.0):.4f} "
        f"ourdef_objective_any_success_rate={results_summary.get('ourdef_objective_any_success_rate', 0.0):.4f} "
        f"ourdef_distinct_success_count_mean={results_summary.get('ourdef_distinct_success_count_mean', 0.0):.4f}",
        flush=True,
    )
    print(
        f"[output] results={output_path} rows={output_rows_count} "
        f"target_compute={target_compute_output_path} rows={output_target_compute_count} "
        f"results_live={live_output_path} "
        f"target_compute_live={live_target_compute_output_path} "
        f"restart_dual_live={live_restart_dual_output_path} "
        f"checkpoint_metrics={checkpoint_metrics_output_path} "
        f"loss_trajectory={loss_trajectory_output_path} "
        f"path_diag={path_diag_output_path} "
        f"manifest={manifest_output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
