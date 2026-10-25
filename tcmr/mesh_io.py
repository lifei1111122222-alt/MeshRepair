from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import trimesh


def load_obj(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.load_mesh(str(path), process=False, maintain_order=True)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(g for g in mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Unsupported mesh at {path}")
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)


def merge_labeled_parts(parts: list[tuple[np.ndarray, np.ndarray, int]], decimals: int = 8):
    """Merge part meshes while preserving a face-to-part label."""
    if not parts:
        raise ValueError("No mesh parts")
    vertices = np.concatenate([p[0] for p in parts], axis=0)
    faces_all, labels = [], []
    offset = 0
    for v, f, label in parts:
        faces_all.append(np.asarray(f, dtype=np.int64) + offset)
        labels.append(np.full(len(f), label, dtype=np.int64))
        offset += len(v)
    faces = np.concatenate(faces_all, axis=0)
    labels = np.concatenate(labels, axis=0)
    # Quantized welding makes interfaces become shared edges while retaining one vertex set.
    scale = 10.0 ** decimals
    keys = np.round(vertices * scale).astype(np.int64)
    _, first, inverse = np.unique(keys, axis=0, return_inverse=True, return_index=True)
    vertices = vertices[first]
    faces = inverse[faces]
    good = np.logical_and.reduce([faces[:, 0] != faces[:, 1], faces[:, 1] != faces[:, 2], faces[:, 0] != faces[:, 2]])
    return vertices, faces[good], labels[good]


def save_mesh_npz(path: str | Path, vertices: np.ndarray, faces: np.ndarray, cue: np.ndarray | None = None,
                  target_loop: np.ndarray | None = None, **meta):
    payload = {"vertices": np.asarray(vertices), "faces": np.asarray(faces)}
    if cue is not None:
        payload["cue"] = np.asarray(cue)
    if target_loop is not None:
        payload["target_loop"] = np.asarray(target_loop)
    if meta:
        payload["meta_json"] = np.array(json.dumps(meta, ensure_ascii=False))
    np.savez_compressed(path, **payload)


def load_mesh_npz(path: str | Path):
    with np.load(path, allow_pickle=False) as z:
        out = {k: z[k] for k in z.files}
    if "meta_json" in out:
        out["meta"] = json.loads(str(out.pop("meta_json")))
    return out
