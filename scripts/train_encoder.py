"""Train the boundary encoder from mesh records or cached observed features."""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tcmr.mesh_io import load_mesh_npz
from tcmr.boundary import extract_boundary_loops, vertex_normals
from tcmr.features import construct_loop_features
from tcmr.encoder import CueConditionedCyclicBoundaryEncoder
from tcmr.evaluation import target_index_from_loop


def _load_feature_cache(path: str | Path, multi_weight: float = 1.0,
                        multi_repeat: int = 1, category_balance: bool = False,
                        category_max_repeat: int = 4, category_power: float = 0.5):
    """Load packed loop features produced by cache_selector_features.py.

    The cache is intentionally converted back to the trainer's small
    ``[(tokens, {scalars: ...}), ...]`` records here, keeping the optimization
    loop and its loss protocol identical to the mesh-reading path.
    """
    with np.load(path, allow_pickle=False) as z:
        tokens = np.asarray(z["tokens"], dtype=np.float32)
        scalars = np.asarray(z["scalars"], dtype=np.float32)
        offsets = np.asarray(z["offsets"], dtype=np.int64)
        targets = np.asarray(z["targets"], dtype=np.int64)
        splits = np.asarray(z["splits"], dtype=np.int8)
        categories = np.asarray(z["categories"])
    if len(offsets) != len(targets) + 1:
        raise ValueError("Invalid feature cache: offsets/targets length mismatch")
    train_counts = {}
    if category_balance:
        for i in range(len(targets)):
            if int(splits[i]) == 0:
                category = str(categories[i])
                train_counts[category] = train_counts.get(category, 0) + 1
    max_count = max(train_counts.values(), default=1)
    samples = []
    val_samples = []
    for i in range(len(targets)):
        lo, hi = int(offsets[i]), int(offsets[i + 1])
        if hi <= lo:
            continue
        feats = [(tokens[j], {"scalars": scalars[j]}) for j in range(lo, hi)]
        y = None if int(targets[i]) < 0 else int(targets[i])
        is_multi = len(feats) > 1
        item = (feats, y, float(multi_weight) if is_multi else 1.0)
        if int(splits[i]) == 1:
            val_samples.append(item)
        elif int(splits[i]) == 0:
            repeat = max(1, int(multi_repeat)) if is_multi else 1
            if category_balance:
                category = str(categories[i])
                count = max(train_counts.get(category, 1), 1)
                factor = (float(max_count) / float(count)) ** float(category_power)
                repeat *= min(max(1, int(round(factor))), max(1, int(category_max_repeat)))
            samples.extend([item] * repeat)
    return samples, val_samples


def _random_null_cue(vertices, rng):
    """Create a cue that is intentionally inconsistent with a damaged mesh.

    The cue is sampled outside an expanded damaged-mesh bounding box, so this
    augmentation never inspects removed-part geometry or target annotations.
    """
    lo, hi = np.asarray(vertices).min(0), np.asarray(vertices).max(0)
    center = (lo + hi) * 0.5
    diag = float(np.linalg.norm(hi - lo)) + 1e-8
    direction = rng.normal(size=3)
    direction /= max(np.linalg.norm(direction), 1e-8)
    distance = rng.uniform(2.5, 5.0) * diag
    scale = rng.uniform(0.15, 0.45) * diag
    return np.concatenate([center + direction * distance, [scale]]).astype(np.float32)


def _perturb_cue(cue, vertices, rng, translation_max=1.0, scale_jitter=1.5):
    """Perturb a positive cue while retaining the same removal target label."""
    cue = np.asarray(cue, dtype=np.float32)
    direction = rng.normal(size=3)
    direction /= max(float(np.linalg.norm(direction)), 1e-8)
    amount = rng.uniform(-float(translation_max), float(translation_max)) * max(float(cue[3]), 1e-8)
    # Log-uniform scale changes avoid over-sampling only enlarged cues.
    log_jitter = np.log(max(float(scale_jitter), 1.0))
    scale = float(cue[3]) * np.exp(rng.uniform(-log_jitter, log_jitter))
    return np.concatenate([cue[:3] + amount * direction, [scale]]).astype(np.float32)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--index", default="data/partnet_tcmr/dataset_index.json"); p.add_argument("--cache", default=None, help="packed selector feature cache; skips mesh/feature extraction"); p.add_argument("--epochs", type=int, default=30); p.add_argument("--seed", type=int, default=0); p.add_argument("--output", default="data/partnet_tcmr/cbe.pt"); p.add_argument("--batch-size", type=int, default=1, help="number of variable-loop samples per optimizer update; 1 preserves legacy updates"); p.add_argument("--init-checkpoint", default=None, help="optional encoder state used for transfer/fine-tuning"); p.add_argument("--device", default=None, help="torch device (default: cuda when available)"); p.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate"); p.add_argument("--category-balance", action="store_true", help="repeat low-frequency training categories using train counts only"); p.add_argument("--category-max-repeat", type=int, default=4); p.add_argument("--category-power", type=float, default=0.5)
    p.add_argument("--null-augment", action="store_true", help="add cue-mismatch null examples using damaged mesh only")
    p.add_argument("--null-weight", type=float, default=0.25, help="relative loss weight for each synthetic null example")
    p.add_argument("--cue-augment", action="store_true", help="add positive cue-perturbed copies of training samples")
    p.add_argument("--cue-translation-max", type=float, default=1.0, help="maximum cue-center shift in original cue-scale units")
    p.add_argument("--cue-scale-jitter", type=float, default=1.5, help="multiplicative log-uniform cue-scale jitter")
    p.add_argument("--pair-weight", type=float, default=0.25,
                   help="weight of target-vs-distractor pairwise ranking loss")
    p.add_argument("--pair-margin", type=float, default=0.0,
                   help="positive margin for pairwise ranking loss")
    p.add_argument("--multi-weight", type=float, default=1.0,
                   help="extra weight for samples with multiple boundary loops")
    p.add_argument("--multi-repeat", type=int, default=1,
                   help="repeat multi-loop training samples this many times")
    p.add_argument("--selection-metric", choices=("overall", "multi"), default="overall",
                   help="validation metric used for checkpoint selection")
    a = p.parse_args()
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    device_name = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    idx = json.loads(Path(a.index).read_text(encoding="utf-8")); model = CueConditionedCyclicBoundaryEncoder().to(device)
    if a.init_checkpoint:
        state = torch.load(a.init_checkpoint, map_location="cpu")
        model.load_state_dict(state)
    opt = torch.optim.Adam(model.parameters(), lr=float(a.lr))
    samples = []; val_samples = []
    if a.cache:
        if a.null_augment:
            raise ValueError("--null-augment requires raw meshes; omit it with --cache")
        samples, val_samples = _load_feature_cache(
            a.cache, multi_weight=a.multi_weight, multi_repeat=a.multi_repeat,
            category_balance=a.category_balance,
            category_max_repeat=a.category_max_repeat,
            category_power=a.category_power)
    for rec in ([] if a.cache else idx["records"]):
        if rec.get("split", "train") not in ("train", "val"): continue
        d = load_mesh_npz(rec["path"]); loops = extract_boundary_loops(d["vertices"], d["faces"]); vn = vertex_normals(d["vertices"], d["faces"]) if loops else None; feats = [construct_loop_features(d["vertices"], d["faces"], l, d["cue"], cached_normals=vn) for l in loops]
        if not feats: continue
        y = target_index_from_loop(loops, d.get("target_loop")); item = (feats, y, float(a.multi_weight) if len(feats) > 1 else 1.0)
        target_list = samples if rec.get("split", "train") == "train" else val_samples
        repeat = max(1, int(a.multi_repeat)) if rec.get("split", "train") == "train" and len(feats) > 1 else 1
        target_list.extend([item] * repeat)
        if a.cue_augment and rec.get("split", "train") == "train":
            aug_cue = _perturb_cue(d["cue"], d["vertices"], rng,
                                   a.cue_translation_max, a.cue_scale_jitter)
            aug_feats = [construct_loop_features(d["vertices"], d["faces"], l,
                                                  aug_cue, cached_normals=vn) for l in loops]
            samples.append((aug_feats, y, float(a.multi_weight) if len(aug_feats) > 1 else 1.0))
        if a.null_augment and rec.get("split", "train") == "train":
            null_cue = _random_null_cue(d["vertices"], rng)
            null_feats = [construct_loop_features(d["vertices"], d["faces"], l, null_cue, cached_normals=vn) for l in loops]
            samples.append((null_feats, None, float(a.null_weight)))
    if not samples: raise RuntimeError("No trainable samples")
    model.train(); best_val = -1.0; best_state = None; patience = 20; stale = 0
    for epoch in range(a.epochs):
        losses=[]
        # Shuffle the expanded training set so oversampled multi-loop records
        # are interleaved with single-loop records rather than forming a block.
        order = [int(j) for j in rng.permutation(len(samples))]
        batch_size = max(1, int(a.batch_size))
        for start in range(0, len(order), batch_size):
            batch = [samples[j] for j in order[start:start + batch_size]]
            if len(batch) == 1:
                feats, y, weight = batch[0]
                tok = torch.tensor(np.stack([x[0] for x in feats]), dtype=torch.float32, device=device); sc = torch.tensor(np.stack([x[1]["scalars"] for x in feats]), dtype=torch.float32, device=device)
                logits = model(tok, sc); target = 0 if y is None else y + 1
                loss = F.cross_entropy(logits[None, :], torch.tensor([target], device=device))
                if y is not None and logits.numel() > 2 and a.pair_weight > 0:
                    pos = logits[target]
                    neg = torch.cat((logits[:target], logits[target + 1:]))
                    # A positive margin makes the ranking term continue to train
                    # until the target logit is separated from each distractor.
                    loss = loss + float(a.pair_weight) * F.softplus(float(a.pair_margin) - (pos - neg)).mean()
                loss = loss * float(weight)
            else:
                # Candidate loops are variable-length.  Flatten them for one
                # convolution pass, then split logits back per sample so each
                # sample keeps its own null hypothesis and target index.
                tok_parts, sc_parts, sizes = [], [], []
                for feats, _, _ in batch:
                    tok_parts.append(torch.tensor(np.stack([x[0] for x in feats]), dtype=torch.float32, device=device))
                    sc_parts.append(torch.tensor(np.stack([x[1]["scalars"] for x in feats]), dtype=torch.float32, device=device))
                    sizes.append(len(feats))
                all_tok = torch.cat(tok_parts, dim=0); all_sc = torch.cat(sc_parts, dim=0)
                all_loop_logits = model.loop_logits(all_tok, all_sc)
                sample_losses, offset = [], 0
                for (feats, y, weight), size, sample_sc in zip(batch, sizes, sc_parts):
                    loop_logits = all_loop_logits[offset:offset + size]; offset += size
                    null_logit = model.null_head(sample_sc.mean(dim=0, keepdim=True)).squeeze(-1)[0]
                    logits = torch.cat((null_logit[None], loop_logits), dim=0)
                    target = 0 if y is None else y + 1
                    item_loss = F.cross_entropy(logits[None, :], torch.tensor([target], device=device))
                    if y is not None and size > 1 and a.pair_weight > 0:
                        pos = logits[target]
                        neg = torch.cat((logits[:target], logits[target + 1:]))
                        item_loss = item_loss + float(a.pair_weight) * F.softplus(float(a.pair_margin) - (pos - neg)).mean()
                    sample_losses.append(item_loss * float(weight))
                loss = torch.stack(sample_losses).mean()
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(float(loss.detach()))
        if val_samples:
            model.eval(); correct = 0; multi_correct = 0; multi_count = 0
            with torch.no_grad():
                # Flatten validation loops into one GPU convolution. The old
                # per-sample forward pass made validation dominate large runs.
                tok_parts = [torch.tensor(np.stack([x[0] for x in feats]), dtype=torch.float32, device=device)
                             for feats, _, _ in val_samples]
                sc_parts = [torch.tensor(np.stack([x[1]["scalars"] for x in feats]), dtype=torch.float32, device=device)
                            for feats, _, _ in val_samples]
                sizes = [len(feats) for feats, _, _ in val_samples]
                all_tok = torch.cat(tok_parts, dim=0); all_sc = torch.cat(sc_parts, dim=0)
                all_loop_logits = model.loop_logits(all_tok, all_sc)
                null_sc = torch.stack([sc.mean(dim=0) for sc in sc_parts], dim=0)
                all_null_logits = model.null_head(null_sc).squeeze(-1)
                offset = 0
                for (feats, y, _), size, null_logit in zip(val_samples, sizes, all_null_logits):
                    loop_logits = all_loop_logits[offset:offset + size]; offset += size
                    pred = int(torch.argmax(torch.cat((null_logit[None], loop_logits))) .item()) - 1
                    correct += int(pred == y)
                    if size > 1:
                        multi_correct += int(pred == y); multi_count += 1
            val_acc = correct / len(val_samples)
            metric = (multi_correct / max(multi_count, 1)) if a.selection_metric == "multi" else val_acc
            if metric > best_val:
                best_val = metric; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; stale = 0
            else: stale += 1
            model.train()
            if stale >= patience: break
        if epoch == 0 or (epoch + 1) % 10 == 0: print(f"epoch={epoch+1} loss={np.mean(losses):.6f}")
    Path(a.output).parent.mkdir(parents=True, exist_ok=True); torch.save(best_state or model.state_dict(), a.output)


if __name__ == "__main__": main()
