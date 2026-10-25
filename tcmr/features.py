from __future__ import annotations

import numpy as np
from .boundary import loop_geometry


def resample_loop(vertices: np.ndarray, loop: np.ndarray, q: int) -> tuple[np.ndarray, np.ndarray]:
    p = vertices[loop]
    nxt = np.roll(p, -1, axis=0); seg = np.linalg.norm(nxt - p, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg)]); total = cumulative[-1]
    targets = np.linspace(0.0, total, q, endpoint=False)
    out = []
    for t in targets:
        i = min(np.searchsorted(cumulative, t, side="right") - 1, len(p) - 1)
        frac = (t - cumulative[i]) / max(seg[i], 1e-12)
        out.append(p[i] * (1 - frac) + nxt[i] * frac)
    return np.asarray(out), targets


def construct_loop_features(vertices: np.ndarray, faces: np.ndarray, loop: np.ndarray, cue: np.ndarray,
                            q: int = 64, cached_normals: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    g = loop_geometry(vertices, faces, loop, cached_normals=cached_normals)
    points, _ = resample_loop(vertices, loop, q)
    # nearest original loop samples provide tangent/normal/curvature tokens.
    d = ((points[:, None, :] - g["points"][None, :, :]) ** 2).sum(-1)
    idx = d.argmin(axis=1)
    center, scale = cue[:3], max(float(cue[3]), 1e-8)
    rel = (points - center) / scale
    dist = np.linalg.norm(points - center, axis=1, keepdims=True) / scale
    # Curvature is inverse length; multiply by the local boundary scale so
    # descriptors remain comparable across PartNet objects of different size.
    curvature_dimless = g["curvature"] * max(g["scale"], 1e-8)
    tokens = np.concatenate([rel, g["tangents"][idx], g["normals"][idx], curvature_dimless[idx, None],
                             g["lengths"][idx, None] / max(g["scale"], 1e-8), dist], axis=1).astype(np.float32)
    bb_min, bb_max = g["points"].min(0), g["points"].max(0)
    scalars = np.array([g["lengths"].sum() / scale, g["planarity"], curvature_dimless.mean(),
                        curvature_dimless.std(), np.prod(np.maximum(bb_max - bb_min, 1e-8)) / scale ** 3], dtype=np.float32)
    return tokens, {"scalars": scalars, "planarity": g["planarity"], "curvature_std": float(g["curvature"].std()),
                    "scale": g["scale"], "points": g["points"], "tangents": g["tangents"], "normals": g["normals"],
                    "curvature": g["curvature"], "lengths": g["lengths"]}
