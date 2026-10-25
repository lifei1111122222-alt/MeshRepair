from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import numpy as np
import torch
from .boundary import extract_boundary_loops, vertex_normals
from .features import construct_loop_features
from .encoder import CueConditionedCyclicBoundaryEncoder
from .patch_init import planar_delaunay_or_fan
from .support import adaptive_support
from .optimizer import optimize_local, triangle_quality
from .validity import check_mesh_validity, patch_retained_intersections, orient_patch_to_boundary
from .decision import cue_cost, joint_decision


@dataclass
class TCMRConfig:
    q: int = 64; top_k: int = 5; eta: float = 3.0; support_a: float = 2.0; support_b: float = 1.0
    alpha: float = 1.0; beta: float = 1.0; gamma: float = 10.0; null_tau: float = 0.0
    score_normalizer: dict | None = None
    # A candidate that creates a very large number of locally intersecting
    # triangles is not a meaningful minimal-change repair.  The validity
    # checker may return ``None`` for large neighborhoods (unknown), so this
    # threshold is applied only when an exact count is available.
    self_intersection_limit: int = 500
    self_intersection_cost: float = 5.0
    # Reject a candidate only when it adds local self intersections relative
    # to the damaged input.  PartNet meshes can contain pre-existing
    # intersections, so absolute counts are not a useful validity criterion.
    reject_new_self_intersections: bool = False
    # A projected polygon can be valid in 2-D while its patch crosses nearby
    # retained geometry in 3-D.  Retry such candidates with the conservative
    # fan initializer and keep it only when the seam-local intersection count
    # decreases.
    retry_fan_on_self_intersection: bool = False
    # Do not export a candidate that introduces a known local defect when an
    # alternative valid candidate or the null hypothesis is available.
    reject_invalid_selected: bool = False
    # When the learned prior is effectively tied, a noisy counterfactual
    # energy must not overturn its deterministic top choice.  The threshold
    # is an absolute probability gap and is applied only to the top two
    # retained candidates.
    prior_gap_guard: float = 0.005
    # Optional transparent cue-rank fusion used on larger cross-category
    # benchmarks.  It is disabled by default for backwards compatibility and
    # can be fitted on validation without touching test labels.
    cue_rank_weight: float = 0.0
    # Optional train-fitted gate for CBE/cue disagreement cases.
    selector_gate: dict | None = None
    # Number of LBFGS iterations used for each counterfactual candidate.
    # The default is the full repair setting; calibration can use a smaller
    # value because it only needs a stable relative repair-cost diagnostic.
    optimization_steps: int = 80
    # Planarity threshold for choosing a planar boundary triangulation. Lower
    # values route more mildly non-planar loops through the fan initializer.
    planarity_threshold: float = 0.03
    # Optional label-free initializer safety selector.  When enabled on
    # sufficiently small loops, the pipeline constructs both the normal
    # advancing-front fan and a forced planar triangulation, then keeps the
    # candidate with the lower seam-local self-intersection count.  It is
    # deliberately disabled by default so historical experiments retain the
    # exact initializer policy.
    initializer_safety_selector: bool = False
    initializer_safety_max_loop: int = 400
    # Optional normal offsets evaluated by the initializer safety selector.
    # Larger offsets can move a fan apex out of retained geometry on folded
    # boundaries.  The selector remains disabled by default and only applies
    # these offsets on its bounded small-loop path.
    initializer_safety_offsets: tuple = (-20.0, -10.0, -5.0, -1.0, -0.5,
                                          0.5, 1.0, 5.0, 10.0, 20.0)
    # Optional label-free multi-objective tie-break.  Self-intersection count
    # remains the primary criterion; seam angle, offset magnitude and triangle
    # quality are only used when that count (and retained collision count)
    # ties.  Disabled by default for exact historical reproducibility.
    initializer_safety_multiobjective: bool = False
    initializer_safety_seam_weight: float = 1.0
    initializer_safety_offset_weight: float = 0.02
    initializer_safety_quality_weight: float = 0.05
    # Fidelity is deliberately strong: collar vertices are retained surface
    # and should move only enough to meet the seam constraints.  The earlier
    # value (10) allowed occasional large excursions on thin, non-planar
    # loops, violating the minimal-change objective.
    weights: dict = field(default_factory=lambda: {"fair": 1.0, "lap": 0.1, "fid": 1000.0, "normal": 1.0, "curv": 0.1, "quality": 1.0})
    # ``residual_variation`` preserves the historical optimizer.  The
    # ``normal_variation`` ablation compares discrete normal variation on
    # both sides of the seam, which is a closer proxy for curvature matching.
    curvature_mode: str = "residual_variation"
    quality_fallback_ratio: float = 0.80
    # If local optimization breaks seam closure that was already satisfied by
    # the deterministic initializer, keep the initializer.  This is a
    # label-free topology safeguard and is separate from the stricter option
    # that rejects every closure failure during candidate ranking.
    revert_if_closure_worse: bool = True
    # Keep local optimization from worsening the collision profile of the
    # deterministic initializer.  The guard is label-free and only compares
    # the candidate patch before and after optimization.
    revert_if_worse_self_intersection: bool = True
    # Optional second-stage guard for the full seam-local self-intersection
    # metric.  The inexpensive retained/patch collision count above does not
    # see intersections among patch triangles themselves.  When enabled, an
    # optimized candidate is compared with its deterministic initializer and
    # reverted if it creates more seam-local self intersections.  Disabled by
    # default for backwards-compatible ablations because it adds one local
    # validity pass for candidates whose optimized mesh is self-intersecting.
    revert_if_worse_patch_self_intersection: bool = False
    collision_retry: bool = True
    # Optional conservative safety policy for ablations: reject a candidate
    # whose initializer already intersects retained geometry, or whose local
    # intersection count increases after optimization.
    reject_initial_collisions: bool = False
    reject_collision_increase: bool = False
    collision_retry_offsets: tuple = (-1.0, -0.5, 0.5, 1.0)
    support_rings: int | None = 3
    collision_cost_weight: float = 0.5
    # Closure is reported for every candidate.  Enabling this guard allows
    # strict diagnostic runs to reject patches that fail to add all seam
    # edges, while the default keeps historical selector scores unchanged.
    reject_closure_failures: bool = False
    closure_tolerance: float = 1e-8
    # Optional validation-locked safety policy.  When enabled, candidate
    # ranking uses the *incremental* seam-local self-intersection count
    # relative to the damaged input.  This keeps inherited PartNet defects
    # from dominating the decision while allowing a top-K alternative to
    # replace a geometrically unsafe top-prior candidate.
    new_self_intersection_weight: float = 0.0
    safety_rerank: bool = False


class TCMRSystem:
    def __init__(self, config: TCMRConfig | None = None, encoder=None, device="cpu"):
        self.config = config or TCMRConfig(); self.device = device
        self.encoder = encoder or CueConditionedCyclicBoundaryEncoder()
        self.encoder.to(device).eval()

    def _priors(self, feature_list):
        if not feature_list:
            return np.empty(0), 1.0
        tokens = torch.tensor(np.stack([x[0] for x in feature_list]), dtype=torch.float32, device=self.device)
        scalars = torch.tensor(np.stack([x[1]["scalars"] for x in feature_list]), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.encoder(tokens, scalars)
            probs = torch.softmax(logits, dim=0).cpu().numpy()
        return probs[1:], float(probs[0])

    @staticmethod
    def _initializer_seam_normal_error(vertices, retained_faces, patch_faces, loop):
        """Mean unoriented seam normal angle for an initializer candidate."""
        loop = np.asarray(loop, dtype=np.int64)
        if len(loop) == 0 or len(retained_faces) == 0 or len(patch_faces) == 0:
            return None
        def edge_map(ff):
            out = {}
            for fi, face in enumerate(np.asarray(ff, dtype=np.int64)):
                for edge in (face[[0, 1]], face[[1, 2]], face[[2, 0]]):
                    out.setdefault(tuple(np.sort(edge).tolist()), fi)
            return out
        def normals(ff):
            tri = np.asarray(vertices, dtype=float)[np.asarray(ff, dtype=np.int64)]
            nn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            return nn / np.maximum(np.linalg.norm(nn, axis=1, keepdims=True), 1e-12)
        bm, pm = edge_map(retained_faces), edge_map(patch_faces)
        bn, pn = normals(retained_faces), normals(patch_faces)
        angles = []
        for a, b in zip(loop, np.roll(loop, -1)):
            key = tuple(sorted((int(a), int(b))))
            if key not in bm or key not in pm:
                continue
            angles.append(float(np.arccos(np.clip(abs(np.dot(bn[bm[key]], pn[pm[key]])), 0.0, 1.0))))
        return float(np.mean(angles)) if angles else None

    def _select_safe_initializer(self, vertices, faces, loop, planarity):
        """Select a lower-risk deterministic patch initializer when enabled.

        The selector is label-free: both candidates are constructed solely
        from the damaged mesh and the selected boundary.  A seam-local
        retained/patch crossing plus patch-internal crossing count (with a
        full self-intersection fallback) is used as the primary cost; the
        retained / patch collision count and mean triangle quality break ties.
        The
        default candidate is first in the ordering, so exact ties preserve
        the historical behaviour.
        """
        cfg = self.config
        default = planar_delaunay_or_fan(
            vertices, loop, planarity, cfg.planarity_threshold)
        info = {
            "enabled": bool(cfg.initializer_safety_selector and
                            len(loop) <= int(cfg.initializer_safety_max_loop)),
            "max_loop": int(cfg.initializer_safety_max_loop),
            "loop_length": int(len(loop)),
            "selected": default[2],
            "candidates": [],
        }
        if not info["enabled"]:
            return default, info

        # Keep the ordinary initializer first for deterministic tie handling.
        candidates = [
            ("default", default, False),
            ("forced_fan", planar_delaunay_or_fan(
                vertices, loop, planarity, threshold=-1.0), False),
            ("forced_planar", planar_delaunay_or_fan(
                vertices, loop, planarity, threshold=float("inf")), False),
        ]
        # Include the configured fan offsets in the same label-free safety
        # comparison.  The ordinary collision-retry branch only runs when
        # the selected initializer already crosses retained geometry; that
        # would miss a planar patch whose retained collision is zero but whose
        # own triangles still overlap.  Duplicate zero offsets are skipped.
        seen_offsets = {0.0}
        offset_values = []
        for raw_offset in cfg.initializer_safety_offsets:
            try:
                offset = float(raw_offset)
            except (TypeError, ValueError):
                continue
            if abs(offset) <= 1e-12 or offset in seen_offsets:
                continue
            seen_offsets.add(offset)
            offset_values.append(offset)
        scored = []
        for label, candidate, is_offset in candidates:
            vv, ff, method, center_idx = candidate
            ff = orient_patch_to_boundary(ff, faces, loop)
            self_count = None
            retained_collision = None
            patch_self = None
            try:
                retained_collision = patch_retained_intersections(
                    vv, faces, ff, loop)
            except Exception:
                retained_collision = None
            # Count crossings among the candidate patch triangles separately;
            # using only the compact retained/patch signal would miss a
            # self-overlapping planar triangulation.  This pass is bounded by
            # the loop-size gate and is much cheaper than checking the full
            # retained neighborhood.
            try:
                patch_self = check_mesh_validity(vv, ff, loop).get(
                    "self_intersections")
            except Exception:
                patch_self = None
            # The retained/patch crossing pass is the same seam-local
            # collision signal used later by the optimizer guard and is much
            # cheaper than the all-pairs self-intersection pass.  Fall back to
            # the full seam-local count only when the compact signal is
            # unavailable (for example, an oversized neighborhood).
            if retained_collision is not None and patch_self is not None:
                self_count = int(retained_collision) + int(patch_self)
            elif retained_collision is not None:
                self_count = int(retained_collision)
            else:
                try:
                    merged = np.concatenate([faces, ff], axis=0)
                    self_count = check_mesh_validity(vv, merged, loop).get(
                        "self_intersections")
                except Exception:
                    self_count = None
            quality = float(np.mean(triangle_quality(vv, ff))) if len(ff) else 0.0
            try:
                offset_magnitude = (abs(float(label.split("fan_offset_", 1)[1]))
                                    if label.startswith("fan_offset_") else 0.0)
            except (ValueError, IndexError):
                offset_magnitude = 0.0
            seam_error = self._initializer_seam_normal_error(vv, faces, ff, loop)
            if cfg.initializer_safety_multiobjective:
                seam_term = float(seam_error) if seam_error is not None else float(np.pi / 2.0)
                tie_cost = (float(cfg.initializer_safety_seam_weight) * seam_term / (np.pi / 2.0) +
                            float(cfg.initializer_safety_offset_weight) * offset_magnitude +
                            float(cfg.initializer_safety_quality_weight) * max(0.0, 1.0 - quality))
                tie_score = (tie_cost, -quality)
            else:
                tie_cost = offset_magnitude
                tie_score = (offset_magnitude, -quality)
            # Unknown self-intersection counts are ranked after every known
            # count, while still allowing a finite fallback collision signal.
            primary = (int(self_count) if self_count is not None else 10**9)
            secondary = (int(retained_collision)
                         if retained_collision is not None else 10**9)
            score = (primary, secondary, *tie_score)
            rec = {
                "label": label,
                "method": method,
                "faces": int(len(ff)),
                "self_intersections": (None if self_count is None else int(self_count)),
                "retained_intersections": (None if retained_collision is None else int(retained_collision)),
                "patch_self_intersections": (None if patch_self is None else int(patch_self)),
                "metric": ("retained_plus_patch_self" if retained_collision is not None
                            and patch_self is not None else "self_intersections"),
                "triangle_quality": quality,
                "offset_magnitude": float(offset_magnitude),
                "initializer_seam_normal_error": (None if seam_error is None else float(seam_error)),
                "score": [int(primary), int(secondary), float(tie_score[0]), float(tie_score[1])],
            }
            info["candidates"].append(rec)
            scored.append((score, len(scored), vv, ff, method, center_idx, is_offset))
        # Explore large normal offsets only when all three basic candidates
        # have a positive seam-local risk.  This preserves minimal change on
        # already-safe loops while still recovering difficult cases such as a
        # planar candidate whose patch triangles overlap internally.
        base_best_primary = min((x[0][0] for x in scored[:3]), default=10**9)
        if base_best_primary > 0 and offset_values:
            for offset in offset_values:
                vv, ff, method, center_idx = planar_delaunay_or_fan(
                    vertices, loop, planarity, threshold=-1.0,
                    fan_offset=offset)
                label = f"fan_offset_{offset:g}"
                ff = orient_patch_to_boundary(ff, faces, loop)
                self_count = None
                retained_collision = None
                patch_self = None
                try:
                    retained_collision = patch_retained_intersections(vv, faces, ff, loop)
                except Exception:
                    retained_collision = None
                try:
                    patch_self = check_mesh_validity(vv, ff, loop).get(
                        "self_intersections")
                except Exception:
                    patch_self = None
                if retained_collision is not None and patch_self is not None:
                    self_count = int(retained_collision) + int(patch_self)
                elif retained_collision is not None:
                    self_count = int(retained_collision)
                else:
                    try:
                        merged = np.concatenate([faces, ff], axis=0)
                        self_count = check_mesh_validity(vv, merged, loop).get(
                            "self_intersections")
                    except Exception:
                        self_count = None
                quality = float(np.mean(triangle_quality(vv, ff))) if len(ff) else 0.0
                offset_magnitude = abs(float(offset))
                seam_error = self._initializer_seam_normal_error(vv, faces, ff, loop)
                if cfg.initializer_safety_multiobjective:
                    seam_term = float(seam_error) if seam_error is not None else float(np.pi / 2.0)
                    tie_cost = (float(cfg.initializer_safety_seam_weight) * seam_term / (np.pi / 2.0) +
                                float(cfg.initializer_safety_offset_weight) * offset_magnitude +
                                float(cfg.initializer_safety_quality_weight) * max(0.0, 1.0 - quality))
                    tie_score = (tie_cost, -quality)
                else:
                    tie_cost = offset_magnitude
                    tie_score = (offset_magnitude, -quality)
                primary = int(self_count) if self_count is not None else 10**9
                secondary = int(retained_collision) if retained_collision is not None else 10**9
                score = (primary, secondary, *tie_score)
                rec = {
                    "label": label,
                    "method": method,
                    "faces": int(len(ff)),
                    "self_intersections": (None if self_count is None else int(self_count)),
                    "retained_intersections": (None if retained_collision is None else int(retained_collision)),
                    "patch_self_intersections": (None if patch_self is None else int(patch_self)),
                    "metric": ("retained_plus_patch_self" if retained_collision is not None and patch_self is not None
                               else "retained_patch_intersections" if retained_collision is not None
                               else "self_intersections"),
                    "triangle_quality": quality,
                    "offset_magnitude": float(offset_magnitude),
                    "initializer_seam_normal_error": (None if seam_error is None else float(seam_error)),
                    "score": [int(primary), int(secondary), float(tie_score[0]), float(tie_score[1])],
                }
                info["candidates"].append(rec)
                scored.append((score, len(scored), vv, ff, method, center_idx, True))
        # ``len(scored)`` is the stable tie-breaker that keeps ``default``.
        eligible = scored if base_best_primary > 0 else scored[:3]
        best_score, best_order, sv, sf, sm, sci, _ = min(
            eligible, key=lambda x: (x[0], x[1]))
        info["selected"] = sm
        info["selected_label"] = info["candidates"][int(best_order)]["label"]
        return (sv, sf, sm, sci), info

    def repair(self, vertices, faces, cue, debug=False, debug_dir=None, sample_id="sample", oracle_target=None):
        loops = extract_boundary_loops(vertices, faces)
        baseline_validity = check_mesh_validity(vertices, faces)
        cached_normals = vertex_normals(vertices, faces) if loops else None
        feats = [construct_loop_features(vertices, faces, loop, cue, self.config.q, cached_normals=cached_normals) for loop in loops]
        priors, null_prior = self._priors(feats)
        cue_costs = np.array([cue_cost(vertices, l, cue) for l in loops])
        # Rank fusion uses the unclipped mean normalized cue distance.  The
        # clipped quadratic ``cue_cost`` remains the robust energy term, but
        # clipping collapses distant loops to ties and would make the
        # validation-fitted rank selector non-deterministic on such cases.
        cue_center, cue_scale = np.asarray(cue[:3], dtype=float), max(float(cue[3]), 1e-8)
        cue_rank_values = np.array([float(np.linalg.norm(vertices[l] - cue_center, axis=1).mean() / cue_scale) for l in loops])
        if len(loops) == 0:
            return {"mesh": (vertices, faces), "selected": None, "loops": [], "null_prior": 1.0, "decision": {"selected": None}}
        # Lower cue cost means better agreement with the edit cue.  The
        # previous reciprocal term accidentally rewarded distant loops.
        prune_scores = -priors + 0.1 * cue_costs
        gate_target = None
        if oracle_target is None and self.config.selector_gate is not None and len(priors) > 1:
            g = self.config.selector_gate
            order_p = np.argsort(-priors); order_d = np.argsort(cue_rank_values)
            po = np.sort(priors)[::-1]; do = np.sort(cue_rank_values)
            gate_feat = np.array([po[0]-po[1], do[1]-do[0], float(-(priors*np.log(np.maximum(priors,1e-8))).sum()), len(priors), float(priors.max()), float(cue_rank_values.min()), float(cue_rank_values.mean()), float(cue_rank_values.std())], dtype=float)
            z = float(((gate_feat - np.asarray(g['mean'])) / np.maximum(np.asarray(g['std']), 1e-6)) @ np.asarray(g['weights']) + float(g['bias']))
            gate_prob = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
            cbe_top = int(order_p[0]); cue_top = int(order_d[0])
            if cbe_top != cue_top and gate_prob >= float(g['threshold']):
                gate_target = cue_top
                # Retain the CBE top candidate as a validity fallback.  The
                # gate is allowed to improve selection, but it must not force
                # a repair that introduces a new mesh defect.
                prune_scores = np.full(len(priors), np.inf, dtype=float)
                prune_scores[gate_target] = -1.0
                prune_scores[cbe_top] = -0.5
        if self.config.cue_rank_weight > 0 and len(priors) > 1:
            cue_order = np.argsort(cue_rank_values, kind="stable")
            cue_rank = np.empty(len(cue_order), dtype=float)
            cue_rank[cue_order] = np.arange(len(cue_order), dtype=float) / max(len(cue_order) - 1, 1)
            prior_rank = np.empty(len(priors), dtype=float)
            prior_order = np.argsort(-priors, kind="stable")
            prior_rank[prior_order] = np.arange(len(prior_order), dtype=float) / max(len(prior_order) - 1, 1)
            prune_scores = prior_rank + float(self.config.cue_rank_weight) * cue_rank
        if oracle_target is None:
            # The selector gate is an explicit policy decision, so its target
            # must be among the candidates that receive a counterfactual
            # repair.  Cue-rank fusion below may overwrite ``prune_scores``;
            # enforce the gate inclusion after the final pruning order is
            # known.  Keep the configured candidate budget whenever possible
            # by replacing its worst slot rather than silently evaluating an
            # unbounded number of candidates.
            budget = max(int(self.config.top_k), 1)
            rank = np.argsort(prune_scores, kind="stable")[:budget]
            if gate_target is not None and int(gate_target) not in set(rank.tolist()):
                if len(rank) >= budget:
                    rank = np.concatenate([rank[:max(budget - 1, 0)], np.asarray([int(gate_target)], dtype=np.int64)])
                else:
                    rank = np.concatenate([rank, np.asarray([int(gate_target)], dtype=np.int64)])
                # Keep diagnostics and downstream ordering deterministic after
                # the explicit inclusion (the gate itself remains selected).
                rank = np.asarray(rank, dtype=np.int64)
        else:
            rank = np.array([int(oracle_target)], dtype=np.int64)
        candidates = []
        repair_costs = np.full(len(loops), np.inf); penalties = np.ones(len(loops))
        prior_order = np.argsort(-priors)
        prune_order = np.argsort(prune_scores, kind="stable")
        diagnostics = {"target_rank": {int(i): int(np.where(prior_order == i)[0][0] + 1) for i in range(len(loops))},
                       "prune_rank": {int(i): int(np.where(prune_order == i)[0][0] + 1) for i in range(len(loops))},
                       "candidate_indices": [int(i) for i in rank], "candidates": {}, "selector_gate_target": gate_target}
        best_mesh = None
        for i in rank:
            loop = loops[i]; meta = feats[i][1]
            (v0, pf, init_method, center_idx), initializer_safety = self._select_safe_initializer(
                vertices, faces, loop, meta["planarity"])
            pf = orient_patch_to_boundary(pf, faces, loop)
            initial_collision = (patch_retained_intersections(v0, faces, pf, loop)
                                 if self.config.collision_retry else None)
            # A fan center at the boundary centroid can sit inside a retained
            # surface. Try a few normal offsets, plus the original projected
            # patch, and keep the lowest seam-local retained/patch collision
            # count with quality as the tie breaker.
            if self.config.collision_retry and initial_collision is not None and initial_collision > 0:
                options = [(v0, pf, init_method, initial_collision)]
                for off in self.config.collision_retry_offsets:
                    vv, ff, mm, _ = planar_delaunay_or_fan(
                        vertices, loop, meta["planarity"], threshold=-1.0,
                        fan_offset=float(off))
                    ff = orient_patch_to_boundary(ff, faces, loop)
                    cc = patch_retained_intersections(vv, faces, ff, loop)
                    if cc is not None:
                        options.append((vv, ff, mm, cc))
                def _option_key(item):
                    vv, ff, mm, cc = item
                    q = float(np.mean(triangle_quality(vv, ff))) if len(ff) else 0.0
                    # A forced fan is the conservative fallback for retained
                    # geometry.  When it ties a projected planar candidate on
                    # the compact collision count, prefer the fan before
                    # using triangle quality; planar quality alone can hide
                    # patch-internal crossings (e.g. 46159_handle).
                    fan_priority = 0 if mm == "advancing_front_fan" else 1
                    return (int(cc) if cc is not None else 10**9,
                            fan_priority, -q)
                v0, pf, init_method, initial_collision = min(options, key=_option_key)
            patch_vertices = np.unique(pf.reshape(-1))
            support, collar, radius, _ = adaptive_support(v0, loop, meta["scale"], meta["planarity"], meta["curvature_std"], self.config.eta, self.config.support_a, self.config.support_b, rings=self.config.support_rings, faces=faces)
            collar = np.asarray([x for x in collar if x < len(vertices)], dtype=np.int64)
            # Keep retained damaged-mesh faces as the seam reference; ``pf``
            # is only the candidate patch topology.
            out, terms, cost = optimize_local(v0, faces, pf, patch_vertices, collar, loop, self.config.weights, steps=self.config.optimization_steps, device=self.device, curvature_mode=self.config.curvature_mode)
            # The soft quality barrier can still allow skinny triangles when
            # seam terms dominate. Preserve a good initializer if optimization
            # materially degrades its mean triangle quality.
            q_init = float(np.mean(triangle_quality(v0, pf))) if len(pf) else 0.0
            q_opt = float(np.mean(triangle_quality(out, pf))) if len(pf) else 0.0
            quality_fallback = bool(q_init > 1e-8 and q_opt < self.config.quality_fallback_ratio * q_init)
            if quality_fallback:
                out = v0.copy()
                cost = float(cost + q_init - q_opt)
            # patch faces need to reference the extended vertex set and can be merged directly.
            repaired_faces = np.concatenate([faces, pf], axis=0)
            validity = check_mesh_validity(out, repaired_faces, loop)
            # A nonzero residual means at least one target boundary edge was
            # not represented twice after patch insertion.  Keep this as a
            # transparent diagnostic; strict/debug runs may opt into using it
            # as a candidate rejection signal through the config flag.
            validity["closure_failed"] = bool(
                np.isfinite(validity.get("closure", 0.0)) and
                float(validity.get("closure", 0.0)) > float(self.config.closure_tolerance)
            )
            closure_reverted = False
            # Optimization can move boundary vertices enough to create a seam
            # gap even when the initializer closed every annotated edge.  Only
            # inspect the initializer when the optimized candidate actually
            # fails closure, keeping the common path inexpensive.
            if (self.config.revert_if_closure_worse and
                    validity.get("closure_failed", False)):
                init_closure_validity = check_mesh_validity(v0, repaired_faces, loop)
                init_closure_failed = bool(
                    np.isfinite(init_closure_validity.get("closure", 0.0)) and
                    float(init_closure_validity.get("closure", 0.0)) > float(self.config.closure_tolerance)
                )
                if not init_closure_failed:
                    out = v0.copy()
                    validity = init_closure_validity
                    validity["closure_failed"] = False
                    closure_reverted = True
            optimized_collision = patch_retained_intersections(out, faces, pf, loop)
            initial_validity = None
            self_intersection_reverted = False
            # The dedicated patch-retained collision signal is cheaper and
            # more specific than the full seam self-intersection pass. Only
            # run the latter when the dedicated count is unavailable.
            need_total_guard = (self.config.revert_if_worse_self_intersection and
                                (initial_collision is None or optimized_collision is None))
            if need_total_guard:
                initial_validity = check_mesh_validity(v0, repaired_faces, loop)
                initial_validity["closure_failed"] = bool(
                    np.isfinite(initial_validity.get("closure", 0.0)) and
                    float(initial_validity.get("closure", 0.0)) > float(self.config.closure_tolerance)
                )
                init_self = initial_validity.get("self_intersections")
                opt_self = validity.get("self_intersections")
                if (init_self is not None and opt_self is not None and
                        int(opt_self) > int(init_self)):
                    out = v0.copy()
                    validity = initial_validity
                    cand_self = validity.get("self_intersections")
                    cost = float(cost + (int(opt_self) - int(init_self)))
                    self_intersection_reverted = True
            if (initial_collision is not None and optimized_collision is not None and
                    int(optimized_collision) > int(initial_collision)):
                collision_delta = int(optimized_collision) - int(initial_collision)
                out = v0.copy()
                validity = check_mesh_validity(out, repaired_faces, loop)
                validity["closure_failed"] = bool(
                    np.isfinite(validity.get("closure", 0.0)) and
                    float(validity.get("closure", 0.0)) > float(self.config.closure_tolerance)
                )
                optimized_collision = initial_collision
                cost = float(cost + collision_delta)
                self_intersection_reverted = True
            # The retained/patch collision guard above cannot detect an
            # intersection between two triangles that both belong to the new
            # patch.  Such crossings are included in ``self_intersections``.
            # Compare the optimized candidate with the deterministic
            # initializer whenever the optional full guard is requested and
            # keep the initializer if optimization makes this metric worse.
            # This remains label-free: both meshes are constructed from the
            # damaged input and the selected boundary only.
            if self.config.revert_if_worse_patch_self_intersection:
                opt_self = validity.get("self_intersections")
                if opt_self is not None:
                    if initial_validity is None:
                        initial_validity = check_mesh_validity(v0, repaired_faces, loop)
                        initial_validity["closure_failed"] = bool(
                            np.isfinite(initial_validity.get("closure", 0.0)) and
                            float(initial_validity.get("closure", 0.0)) > float(self.config.closure_tolerance)
                        )
                    init_self = initial_validity.get("self_intersections")
                    if (init_self is not None and int(opt_self) > int(init_self)):
                        out = v0.copy()
                        validity = initial_validity
                        optimized_collision = initial_collision
                        cost = float(cost + (int(opt_self) - int(init_self)))
                        self_intersection_reverted = True
            # Compare against the same seam-local diagnostic on the damaged
            # mesh.  This prevents inherited assembly defects from being
            # attributed to the repair while catching patch triangles that
            # newly cross retained surface triangles.
            # Avoid a second quadratic seam check when the candidate has no
            # intersections at all; a zero candidate cannot introduce a new
            # crossing.  This keeps the validity guard affordable on large
            # PartNet meshes.
            cand_self = validity.get("self_intersections")
            baseline_local = (check_mesh_validity(vertices, faces, loop)
                              if cand_self is not None and int(cand_self) > 0
                              else None)
            # PartNet assemblies may already contain duplicated/coincident interfaces.
            # Penalize only defects introduced by the candidate relative to the damaged input.
            introduced = (validity["non_manifold_edges"] > baseline_validity["non_manifold_edges"] or
                          validity["degenerate_faces"] > baseline_validity["degenerate_faces"] or
                          validity["inverted_faces"] > baseline_validity["inverted_faces"] or
                          (self.config.reject_closure_failures and validity.get("closure_failed", False)))
            # Preserve the explicit collision policy after any retry/fallback.
            # This must be applied after the retry branch because that branch
            # recomputes validity and otherwise could erase the guard signal.
            collision_guard = bool(
                self.config.reject_initial_collisions and initial_collision is not None and int(initial_collision) > 0
            ) or bool(
                self.config.reject_collision_increase and
                initial_collision is not None and optimized_collision is not None and
                int(optimized_collision) > int(initial_collision)
            )
            validity["collision_guard_rejected"] = collision_guard
            introduced = introduced or collision_guard
            collision_guard = bool(
                self.config.reject_initial_collisions and initial_collision is not None and int(initial_collision) > 0
            ) or bool(
                self.config.reject_collision_increase and
                initial_collision is not None and optimized_collision is not None and
                int(optimized_collision) > int(initial_collision)
            )
            validity["collision_guard_rejected"] = collision_guard
            if collision_guard:
                introduced = True
            base_self = baseline_local.get("self_intersections") if baseline_local is not None else None
            new_self = (cand_self is not None and base_self is not None and
                        int(cand_self) > int(base_self))
            validity["baseline_self_intersections"] = base_self
            validity["introduced_self_intersections"] = bool(new_self)
            new_self_delta = (max(int(cand_self) - int(base_self), 0)
                              if cand_self is not None and base_self is not None
                              else 0)
            if (self.config.retry_fan_on_self_intersection and new_self and
                    init_method == "planar_delaunay"):
                # A fan is deliberately used only as a validity fallback; the
                # learned selector and normal initializer remain unchanged for
                # ordinary candidates.
                fv0, fpf, fmethod, _ = planar_delaunay_or_fan(
                    vertices, loop, meta["planarity"], threshold=-1.0)
                fpf = orient_patch_to_boundary(fpf, faces, loop)
                fpv = np.unique(fpf.reshape(-1))
                fsupport, fcollar, fradius, _ = adaptive_support(
                    fv0, loop, meta["scale"], meta["planarity"],
                    meta["curvature_std"], self.config.eta,
                    self.config.support_a, self.config.support_b,
                    rings=self.config.support_rings, faces=faces)
                fcollar = np.asarray([x for x in fcollar if x < len(vertices)], dtype=np.int64)
                fout, fterms, fcost = optimize_local(
                    fv0, faces, fpf, fpv, fcollar, loop, self.config.weights,
                    steps=self.config.optimization_steps, device=self.device,
                    curvature_mode=self.config.curvature_mode)
                frepaired = np.concatenate([faces, fpf], axis=0)
                fvalidity = check_mesh_validity(fout, frepaired, loop)
                fself = fvalidity.get("self_intersections")
                # Prefer a lower local intersection count; retain the original
                # projected patch when the retry is not an improvement.
                init_limit = (initial_validity.get("self_intersections")
                              if initial_validity is not None else None)
                if (fself is not None and (cand_self is None or
                        int(fself) < int(cand_self)) and
                        (init_limit is None or int(fself) <= int(init_limit))):
                    v0, pf, init_method = fv0, fpf, fmethod
                    out, terms, cost = fout, fterms, fcost
                    repaired_faces, validity = frepaired, fvalidity
                    validity["closure_failed"] = bool(
                        np.isfinite(validity.get("closure", 0.0)) and
                        float(validity.get("closure", 0.0)) > float(self.config.closure_tolerance)
                    )
                    patch_vertices = fpv; support = fsupport; collar = fcollar
                    radius = fradius; cand_self = fself
                    new_self = (base_self is not None and
                                int(cand_self) > int(base_self))
                    validity["baseline_self_intersections"] = base_self
                    validity["introduced_self_intersections"] = bool(new_self)
                    new_self_delta = (max(int(cand_self) - int(base_self), 0)
                                      if cand_self is not None and base_self is not None
                                      else 0)
            # Recompute the non-manifold/degenerate comparison after a fan
            # retry because the fallback may use a different face count.
            introduced = (validity["non_manifold_edges"] > baseline_validity["non_manifold_edges"] or
                          validity["degenerate_faces"] > baseline_validity["degenerate_faces"] or
                          validity["inverted_faces"] > baseline_validity["inverted_faces"] or
                          (self.config.reject_closure_failures and validity.get("closure_failed", False)))
            if self.config.reject_new_self_intersections:
                introduced = introduced or new_self
            validity["baseline_non_manifold_edges"] = baseline_validity["non_manifold_edges"]
            validity["introduced_defect"] = bool(introduced)
            penalty = 1.0 if introduced else 0.0
            validity["new_self_intersection_delta"] = int(new_self_delta)
            # Self-intersections are measured in the compact seam
            # neighborhood.  Existing PartNet meshes can contain defects, so
            # only an excessive, exactly-counted value is rejected here; an
            # unknown value remains unknown and is not silently treated as a
            # pass or a failure.
            self_x = validity.get("self_intersections")
            if (self_x is not None and
                    int(self_x) > int(self.config.self_intersection_limit)):
                cost += float(self.config.self_intersection_cost) * (
                    float(int(self_x) - int(self.config.self_intersection_limit)) /
                    max(float(self.config.self_intersection_limit), 1.0))
                validity["excess_self_intersections"] = int(self_x) - int(self.config.self_intersection_limit)
            collision_cost = (float(self.config.collision_cost_weight) *
                              float(np.log1p(max(int(optimized_collision or 0), 0))))
            safety_cost = float(self.config.new_self_intersection_weight) * float(new_self_delta)
            repair_costs[i] = cost + collision_cost + safety_cost + 0.01 * len(support) / max(len(vertices), 1)
            penalties[i] = penalty
            diagnostics["candidates"][int(i)] = {"prior": float(priors[i]), "cue_cost": float(cue_costs[i]), "repair_cost": float(repair_costs[i]), "safety_cost": safety_cost, "validity": validity, "energy": terms, "support_size": int(len(support)), "init_method": init_method, "radius": radius, "initial_quality": q_init, "optimized_quality": q_opt, "quality_fallback": quality_fallback, "closure_reverted": closure_reverted, "self_intersection_reverted": self_intersection_reverted, "initial_patch_retained_intersections": initial_collision, "optimized_patch_retained_intersections": optimized_collision, "initializer_safety": initializer_safety}
            candidates.append((i, out, repaired_faces, validity))
        decision = joint_decision(priors[rank], cue_costs[rank], repair_costs[rank], penalties[rank], self.config.alpha, self.config.beta, self.config.gamma, null_prior, self.config.null_tau, self.config.score_normalizer)
        # On the 2000-sample validation benchmark, the locked prior+cue rank
        # fusion is more stable than the counterfactual repair-cost terms.
        # Keep the latter in diagnostics for ablations, but use rank fusion as
        # the final selector when explicitly enabled.  This is the prescribed
        # fallback when repairability does not add stable target-selection
        # gain.  The null hypothesis can still reject all repairs.
        if oracle_target is None and self.config.cue_rank_weight > 0 and len(rank):
            n_all = len(priors)
            prior_order_all = np.argsort(-priors, kind="stable")
            cue_order_all = np.argsort(cue_rank_values, kind="stable")
            prior_rank_all = np.empty(n_all, dtype=float); cue_rank_all = np.empty(n_all, dtype=float)
            prior_rank_all[prior_order_all] = np.arange(n_all, dtype=float) / max(n_all - 1, 1)
            cue_rank_all[cue_order_all] = np.arange(n_all, dtype=float) / max(n_all - 1, 1)
            rank_scores = prior_rank_all[rank] + float(self.config.cue_rank_weight) * cue_rank_all[rank]
            decision["rank_fusion_scores"] = rank_scores.tolist()
            if float(decision.get("null_score", np.inf)) <= float(np.min(rank_scores)):
                decision["selected"] = None
            else:
                decision["selected"] = int(np.argmin(rank_scores))
        order_scores = decision.get("rank_fusion_scores", decision.get("scores", []))
        joint_order = rank[np.argsort(np.asarray(order_scores), kind="stable")] if len(rank) else np.empty(0, dtype=np.int64)
        # Counterfactual terms are auxiliary evidence.  If the learned prior
        # cannot distinguish its two best candidates, keep its top choice
        # instead of letting small normalized geometry-cost fluctuations flip
        # the target.  This guard is fitted independently of test labels and
        # leaves clear prior wins and genuinely ambiguous multi-loop cases to
        # the joint score.
        guarded = False
        if oracle_target is None and self.config.cue_rank_weight <= 0 and len(rank) >= 2 and self.config.prior_gap_guard > 0:
            porder = rank[np.argsort(-priors[rank], kind="stable")]
            if float(priors[porder[0]] - priors[porder[1]]) <= float(self.config.prior_gap_guard):
                top_idx = int(porder[0])
                decision["selected"] = int(np.where(rank == top_idx)[0][0])
                decision["guarded_prior"] = top_idx
                guarded = True
        # A validation-locked selector gate is an explicit target-selection
        # policy.  Preserve its decision through the downstream repair-cost and
        # null-hypothesis bookkeeping; otherwise the gate's benefit can be
        # silently undone after only one candidate was optimized.
        if oracle_target is None and gate_target is not None:
            decision["selected"] = 0
            decision["gate_forced"] = True
        selected = int(oracle_target) if oracle_target is not None else (gate_target if gate_target is not None else (None if decision["selected"] is None else int(rank[decision["selected"]])))
        if oracle_target is None and gate_target is not None:
            gate_candidate = next((x for x in candidates if x[0] == gate_target), None)
            cbe_top = int(np.argsort(-priors, kind="stable")[0])
            cbe_candidate = next((x for x in candidates if x[0] == cbe_top), None)
            if gate_candidate is not None and cbe_candidate is not None and gate_candidate[3].get("introduced_defect", False):
                selected = cbe_top
                diagnostics["gate_fallback"] = "introduced_defect"
        # Safety-aware top-K reranking.  The learned prior remains the
        # primary selector; this branch only activates when its selected
        # candidate creates a measured new local defect and a counterfactual
        # alternative is available.  It therefore preserves target accuracy
        # on clean candidates while making the safety tradeoff explicit and
        # validation-tunable.
        if (oracle_target is None and self.config.safety_rerank and
                selected is not None):
            selected_item = next((x for x in candidates if x[0] == selected), None)
            selected_bad = bool(selected_item is not None and
                                (selected_item[3].get("introduced_defect", False) or
                                 selected_item[3].get("introduced_self_intersections", False)))
            if selected_bad:
                ordered = ([int(i) for i in joint_order.tolist()]
                           if len(joint_order) else [int(i) for i in rank.tolist()])
                safe_ids = []
                for candidate_id in ordered:
                    item = next((x for x in candidates if x[0] == candidate_id), None)
                    if item is not None and not (item[3].get("introduced_defect", False) or
                                                 item[3].get("introduced_self_intersections", False)):
                        safe_ids.append(candidate_id)
                if safe_ids:
                    selected = int(safe_ids[0])
                    diagnostics["safety_fallback"] = "next_no_new_local_defect"
                else:
                    diagnostics["safety_fallback"] = "no_safe_top_k_candidate"
        if (oracle_target is None and self.config.reject_invalid_selected and
                selected is not None):
            selected_item = next((x for x in candidates if x[0] == selected), None)
            selected_invalid = bool(selected_item is not None and
                                    selected_item[3].get("introduced_defect", False))
            if selected_invalid:
                ordered = ([int(i) for i in joint_order.tolist()]
                           if len(joint_order) else [int(i) for i in rank.tolist()])
                valid_ids = []
                for candidate_id in ordered:
                    item = next((x for x in candidates if x[0] == candidate_id), None)
                    if item is not None and not item[3].get("introduced_defect", False):
                        valid_ids.append(candidate_id)
                if valid_ids:
                    selected = valid_ids[0]
                    diagnostics["validity_fallback"] = "next_valid_candidate"
                else:
                    selected = None
                    diagnostics["validity_fallback"] = "null_hypothesis"
        if selected is not None:
            for i, out, repaired_faces, validity in candidates:
                if i == selected:
                    # ``repaired_faces`` already contains the retained faces
                    # and the candidate patch.  Appending ``faces`` again here
                    # duplicates the complete retained surface and creates
                    # artificial non-manifold edges in the final export.
                    best_mesh = (out, repaired_faces); break
        if best_mesh is None:
            best_mesh = (vertices, faces)
        diagnostics.update({"selected": selected, "null_prior": null_prior, "decision": decision,
                            "prior_gap_guarded": guarded,
                            "joint_order": joint_order.astype(int).tolist(),
                            "joint_rank": {int(i): int(np.where(joint_order == i)[0][0] + 1) for i in joint_order},
                            "num_loops": len(loops), "baseline_validity": baseline_validity})
        if debug and debug_dir is not None:
            out_dir = Path(debug_dir); out_dir.mkdir(parents=True, exist_ok=True)
            for ci, cand in diagnostics["candidates"].items():
                np.savez_compressed(out_dir / f"{sample_id}_candidate_{ci}.npz", vertices=next(x[1] for x in candidates if x[0] == ci), faces=next(x[2] for x in candidates if x[0] == ci))
            def _json_default(obj):
                if isinstance(obj, np.ndarray): return obj.tolist()
                if isinstance(obj, (np.floating, np.integer)): return obj.item()
                raise TypeError(type(obj).__name__)
            (out_dir / f"{sample_id}_diagnostics.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
        return {"mesh": best_mesh, "selected": selected, "loops": loops, "priors": priors, "null_prior": null_prior, "diagnostics": diagnostics}
