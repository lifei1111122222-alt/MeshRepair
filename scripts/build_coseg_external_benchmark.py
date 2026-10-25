"""Build a fixed, source-disjoint COSEG external benchmark.

The benchmark uses only official COSEG meshes and face labels during data
construction.  Inference receives vertices, faces and a cue; target labels
remain in the protocol for scoring only.  One source mesh contributes at most
one case, and previously used public-category source IDs are excluded.
"""
from __future__ import annotations

import collections
import hashlib
import json
import sys
import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tcmr.boundary import extract_boundary_loops

OLD_PROTOCOL = ROOT / "data" / "coseg_public_distractor" / "protocol.json"
OUT = ROOT / "data" / "coseg_external_benchmark"
SOURCES = {
    "candelabra": (ROOT / "data" / "coseg_external_raw" / "Candelabra", 16),
    "four_legged": (ROOT / "data" / "coseg_external_raw" / "Four-legged", 10),
    "goblets": (ROOT / "data" / "coseg_external_raw" / "Goblets", 12),
    "guitars": (ROOT / "data" / "coseg_external_raw" / "Guitars", 41),
}
EXPANSION_SOURCES = {
    "tele_aliens": (ROOT / "data" / "coseg_external_raw" / "Tele-aliens", 35),
    "large_vases": (ROOT / "data" / "coseg_external_raw" / "Large-Vases", 35),
    "large_chairs": (ROOT / "data" / "coseg_external_raw" / "Large-Chairs", 35),
}
HOLDOUT_SOURCES = {
    # The small COSEG categories are already fully represented by earlier
    # protocols after raw-mesh hash deduplication.  The unused tail of the
    # official large Vases set supplies an independent confirmation pool.
    "holdout_large_vases": (ROOT / "data" / "coseg_external_raw" / "Large-Vases", 50),
}
CROSS_CATEGORY_SOURCES = {
    "cross_large_chairs": ROOT / "data/coseg_external_raw/Large-Chairs",
    "cross_large_vases": ROOT / "data/coseg_external_raw/Large-Vases",
    "cross_tele_aliens": ROOT / "data/coseg_external_raw/Tele-aliens",
}


def historical_source_hashes() -> tuple[set[str], list[dict]]:
    """Read source references in existing protocols; never inspect predictions."""
    hashes, inventory, paths_seen = set(), [], set()
    for folder in (ROOT / "data", ROOT / "experiments"):
        for path in sorted(folder.rglob("protocol.json")):
            try:
                protocol = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            before = len(hashes)
            cases, sources = protocol.get("cases", []), protocol.get("sources", [])
            for row in (cases if isinstance(cases, list) else []) + (sources if isinstance(sources, list) else []):
                if not isinstance(row, dict):
                    continue
                for key in ("source_mesh_sha256", "raw_mesh_sha256", "raw_sha256"):
                    value = row.get(key)
                    if isinstance(value, str) and len(value) == 64:
                        hashes.add(value)
                for key in ("raw_mesh", "raw_path", "original_path", "source_mesh_path"):
                    value = row.get(key)
                    if not value or str(value) in paths_seen:
                        continue
                    source = Path(value)
                    paths_seen.add(str(value))
                    if source.is_file():
                        hashes.add(sha256(source))
            if len(hashes) > before:
                inventory.append({"protocol": str(path.resolve()), "sha256": sha256(path)})
    return hashes, inventory


def disk_boundary(faces: np.ndarray) -> np.ndarray | None:
    """Single disk boundary using deletion topology, independent of inference."""
    from scripts.local_repair_topology import edge_counts, patch_disk
    if not patch_disk(faces):
        return None
    edges = [e for e, n in edge_counts(faces).items() if n == 1]
    adjacent = collections.defaultdict(list)
    for a, b in edges:
        adjacent[a].append(b)
        adjacent[b].append(a)
    loop = [min(adjacent)]
    while len(loop) < len(adjacent):
        nxt = sorted(set(adjacent[loop[-1]]) - set(loop))
        if not nxt:
            return None
        loop.append(nxt[0])
    return np.asarray(loop, np.int64)


def semantic_target_with_surface_distractor(category, source_root, path, stratum, ordinal):
    """Remove a whole labelled part and a separate retained-surface disk."""
    import trimesh
    from scripts.local_repair_topology import edge_counts
    seg_paths = list((source_root / "gt").rglob(f"{path.stem}.seg"))
    if len(seg_paths) != 1:
        return None
    vertices, faces = read_off(path)
    labels = np.fromstring(seg_paths[0].read_text(encoding="utf-8"), sep=" ", dtype=np.int64)
    if len(labels) != len(faces) or any(n != 2 for n in edge_counts(faces).values()):
        return None
    for target_label in sorted(set(map(int, labels))):
        target_removed = np.flatnonzero(labels == target_label)
        target_loop = disk_boundary(faces[target_removed])
        if target_loop is None:
            continue
        retained_ids = np.flatnonzero(labels != target_label)
        retained_faces = faces[retained_ids]
        if len(retained_faces) < 100:
            continue
        target_vertices = set(faces[target_removed].ravel().tolist())
        target_length = perimeter(vertices, target_loop)
        mesh = trimesh.Trimesh(vertices=vertices, faces=retained_faces, process=False)
        adjacency = collections.defaultdict(list)
        for a, b in mesh.face_adjacency:
            adjacency[int(a)].append(int(b)); adjacency[int(b)].append(int(a))
        centers = vertices[retained_faces].mean(axis=1)
        target_center = vertices[np.unique(faces[target_removed])].mean(axis=0)
        order = np.lexsort((np.arange(len(centers)), -np.linalg.norm(centers-target_center, axis=1)))
        # Fixed, distributed seed ranks avoid searching adaptively on model output.
        seeds = order[np.unique(np.linspace(0, min(len(order)-1, 500), 24).astype(int))]
        for count in (12, 24, 48, 96, 192, 384, 768):
            if count >= len(retained_faces) // 2:
                continue
            for seed in seeds:
                queue, seen = [int(seed)], {int(seed)}
                for face in queue:
                    if len(queue) >= count:
                        break
                    for neighbor in sorted(adjacency[face]):
                        if neighbor not in seen:
                            seen.add(neighbor); queue.append(neighbor)
                            if len(queue) >= count:
                                break
                if len(queue) < count:
                    continue
                distractor_removed = retained_ids[np.asarray(queue[:count], np.int64)]
                if target_vertices.intersection(faces[distractor_removed].ravel().tolist()):
                    continue
                distractor_loop = disk_boundary(faces[distractor_removed])
                if distractor_loop is None:
                    continue
                distractor_length = perimeter(vertices, distractor_loop)
                ratio = target_length / distractor_length
                observed = "smaller" if ratio < .8 else "larger" if ratio > 1.25 else "similar"
                if observed != stratum:
                    continue
                keep = np.ones(len(faces), bool)
                keep[target_removed] = False; keep[distractor_removed] = False
                used = np.unique(faces[keep])
                remap = np.full(len(vertices), -1, np.int64); remap[used] = np.arange(len(used))
                damaged_v, damaged_f = vertices[used], remap[faces[keep]]
                loops = extract_boundary_loops(damaged_v, damaged_f)
                if len(loops) != 2:
                    continue
                mapped_target, mapped_distractor = remap[target_loop], remap[distractor_loop]
                if {frozenset(x.tolist()) for x in loops} != {frozenset(mapped_target.tolist()), frozenset(mapped_distractor.tolist())}:
                    continue
                # Cue stress is fixed before inference and measured along the
                # line between the removed target region and distractor center.
                jitter = (0., .2, .4, .6)[ordinal % 4]
                distractor_center = vertices[np.unique(faces[distractor_removed])].mean(axis=0)
                cue = cue_for_label(vertices, faces, labels, target_label)
                cue[:3] += jitter * (distractor_center-target_center)
                return {"sample_id": f"coseg_cross_{category}_{path.stem}", "category": category,
                        "source_model_id": f"{category}:{path.stem}", "raw_mesh": str(path.resolve()),
                        "raw_segmentation": str(seg_paths[0].resolve()), "source_mesh_sha256": sha256(path),
                        "target_label": target_label, "target_kind": "whole_semantic_label_region",
                        "distractor_kind": "connected_surface_disk", "target_size_stratum": stratum,
                        "target_perimeter": target_length, "distractor_perimeter": distractor_length,
                        "target_to_distractor_perimeter_ratio": ratio,
                        "cue_offset_fraction_of_center_separation": jitter, "cue": cue.tolist(),
                        "target_loop_vertices": mapped_target.tolist(), "distractor_loop_vertices": mapped_distractor.tolist(),
                        "removed_target_face_ids": target_removed.tolist(),
                        "removed_distractor_face_ids": distractor_removed.tolist(),
                        "retained_original_vertex_ids": used.tolist(), "true_interface_verified": True,
                        "vertices": damaged_v, "faces": damaged_f}
    return None


def build_cross_category(out: Path, per_category: int = 20) -> None:
    if out.exists():
        raise FileExistsError(f"refusing to overwrite frozen inputs: {out}")
    excluded, inventory = historical_source_hashes()
    selected, seen, availability = [], set(excluded), {}
    # Exactly 8 small, 8 large, 4 similar target interfaces per category at N=20.
    strata = (["smaller", "larger", "similar", "smaller", "larger"] * ((per_category+4)//5))[:per_category]
    for category, source_root in CROSS_CATEGORY_SOURCES.items():
        paths = sorted((source_root/"shapes").rglob("*.off"), key=lambda p: (int(p.stem) if p.stem.isdigit() else 10**9, p.stem))
        unused = [p for p in paths if sha256(p) not in seen]
        availability[category] = {"raw_sources": len(paths), "unused_sources": len(unused), "chosen": 0}
        for ordinal, stratum in enumerate(strata):
            chosen = None
            for path in unused:
                if sha256(path) in seen:
                    continue
                chosen = semantic_target_with_surface_distractor(category, source_root, path, stratum, ordinal)
                if chosen is not None:
                    break
            if chosen is None:
                raise RuntimeError(f"No unused semantic target + disk distractor for {category}/{stratum}; built {len(selected)} candidates in memory, no formal inputs written")
            selected.append(chosen); seen.add(chosen["source_mesh_sha256"])
            availability[category]["chosen"] += 1
            print(json.dumps({"prepared": len(selected), "source": chosen["source_model_id"], "stratum": stratum,
                              "ratio": chosen["target_to_distractor_perimeter_ratio"]}), flush=True)
    cases_dir = out / "cases"; cases_dir.mkdir(parents=True)
    for item in selected:
        output = cases_dir / f"{item['sample_id']}.npz"
        np.savez_compressed(output, vertices=item.pop("vertices"), faces=item.pop("faces"), cue=np.asarray(item["cue"], float))
        item.update(input_path=str(output.resolve()), input_sha256=sha256(output))
    protocol = {"schema": "coseg_independent_cross_category_semantic_target", "source": "Official COSEG Large-Chairs, Large-Vases, Tele-aliens; unused raw source hashes",
                "source_url": "https://www.yunhaiwang.net/public_html/ssl/ssd.htm", "n_cases": len(selected),
                "n_sources": len(selected), "independent_unit": "source_model_id", "category_counts": {k:v["chosen"] for k,v in availability.items()},
                "selection": "Fixed numeric source order; first structurally eligible unused whole-semantic-label target plus separate surface disk; 8 smaller, 8 larger, 4 similar interfaces per category; no predictions used",
                "label_assignment": "Target is the boundary of the removed entire semantic label, distractor is the separately recorded connected face disk; deletion records are score-only",
                "strict_interface_check": True, "target_labels_read": False, "source_availability": availability,
                "excluded_source_hashes": sorted(excluded), "prior_protocols": inventory,
                "cue_stress": {"offset_fractions": [0., .2, .4, .6], "definition": "cue center shifted toward distractor removed-face center by fixed fraction of center separation; 0.6 is severe ambiguity stress, report separately"},
                "generation_implementation_sha256": sha256(Path(__file__)), "cases": selected}
    (out/"protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps({"complete": True, "n_cases": len(selected), "category_counts": protocol["category_counts"]}), flush=True)


def read_off(path: Path) -> tuple[np.ndarray, np.ndarray]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    if lines[0].lower() != "off":
        raise ValueError(f"not OFF: {path}")
    nv, nf, _ = map(int, lines[1].split()[:3])
    vertices = np.asarray([[float(x) for x in lines[2 + i].split()[:3]]
                           for i in range(nv)], dtype=float)
    faces = []
    for i in range(nf):
        row = [int(x) for x in lines[2 + nv + i].split()]
        if row[0] != 3:
            raise ValueError(f"non-triangular face in {path}")
        faces.append(row[1:4])
    return vertices, np.asarray(faces, dtype=np.int64)


def edge_label_map(faces: np.ndarray, labels: np.ndarray) -> dict[tuple[int, int], set[int]]:
    result: dict[tuple[int, int], set[int]] = collections.defaultdict(set)
    for face, label in zip(faces, labels):
        for a, b in zip(face, np.roll(face, -1)):
            result[tuple(sorted((int(a), int(b))))].add(int(label))
    return result


def loop_label(loop: np.ndarray, used: np.ndarray,
               edge_labels: dict[tuple[int, int], set[int]],
               removed: set[int]) -> int | None:
    original = np.asarray(used[np.asarray(loop, dtype=np.int64)], dtype=np.int64)
    votes: collections.Counter[int] = collections.Counter()
    for a, b in zip(original, np.roll(original, -1)):
        adjacent = edge_labels.get(tuple(sorted((int(a), int(b)))), set())
        for label in adjacent & removed:
            votes[int(label)] += 1
    if not votes:
        return None
    label, count = votes.most_common(1)[0]
    return int(label) if count >= max(3, len(loop) // 2) else None


def perimeter(vertices: np.ndarray, loop: np.ndarray) -> float:
    points = vertices[np.asarray(loop, dtype=np.int64)]
    return float(np.linalg.norm(points - np.roll(points, -1, axis=0), axis=1).sum())


def cue_for_label(vertices: np.ndarray, faces: np.ndarray, labels: np.ndarray,
                  label: int) -> np.ndarray:
    ids = np.unique(faces[labels == label].ravel())
    points = vertices[ids]
    center = points.mean(axis=0)
    scale = max(float(np.linalg.norm(points.max(axis=0) - points.min(axis=0))), 1e-8)
    return np.r_[center, scale]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def used_source_ids() -> set[str]:
    old = json.loads(OLD_PROTOCOL.read_text(encoding="utf-8"))
    used = set()
    for row in old.get("cases", []):
        used.add(str(row["source_model_id"]))
    return used


def used_source_hashes(include_expansion: bool = False) -> set[str]:
    """Collect raw source hashes recorded by earlier COSEG protocols.

    Expansion selection is source-disjoint by content, rather than relying on
    category/name strings that can be reused by the large and small sets.
    Missing legacy paths are simply ignored; their source IDs remain covered
    by ``used_source_ids``.
    """
    hashes: set[str] = set()
    for protocol_path in (ROOT / "data").rglob("protocol.json"):
        # Do not treat a previously generated expansion as an earlier source
        # pool when rebuilding it after a failed/partial run.
        if not include_expansion and "coseg_external_expansion" in protocol_path.parts:
            continue
        try:
            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for row in protocol.get("cases", []):
            recorded = row.get("source_mesh_sha256") or row.get("raw_mesh_sha256")
            if isinstance(recorded, str) and len(recorded) == 64:
                hashes.add(recorded)
            for key in ("raw_mesh", "original_path", "raw_path", "mesh_path"):
                value = row.get(key)
                if not value:
                    continue
                path = Path(value)
                if path.exists() and path.is_file():
                    try:
                        hashes.add(sha256(path))
                    except OSError:
                        pass
                    break
    return hashes


def candidate_for_mesh(category: str, source_root: Path, path: Path,
                       used_ids: set[str], used_hashes: set[str] | None = None,
                       strict: bool = False) -> dict | None:
    source_id = f"{category}:{path.stem}"
    if source_id in used_ids:
        return None
    mesh_sha256 = sha256(path)
    if used_hashes is not None and mesh_sha256 in used_hashes:
        return None
    seg_paths = list((source_root / "gt").rglob(f"{path.stem}.seg"))
    if len(seg_paths) != 1:
        return None
    seg_path = seg_paths[0]
    vertices, faces = read_off(path)
    labels = np.asarray([int(x) for x in seg_path.read_text(encoding="utf-8").split()],
                        dtype=np.int64)
    if len(labels) != len(faces):
        return None
    unique = sorted(set(map(int, labels)))
    edge_labels = edge_label_map(faces, labels)
    for a_i, a in enumerate(unique):
        for b in unique[a_i + 1:]:
            keep = (labels != a) & (labels != b)
            if int(keep.sum()) < 4:
                continue
            kept_vertices = np.unique(faces[keep].ravel())
            remap = np.full(len(vertices), -1, dtype=np.int64)
            remap[kept_vertices] = np.arange(len(kept_vertices), dtype=np.int64)
            damaged_vertices = vertices[kept_vertices]
            damaged_faces = remap[faces[keep]]
            try:
                loops = extract_boundary_loops(damaged_vertices, damaged_faces)
            except Exception:
                continue
            if len(loops) != 2 or any(len(loop) < 3 for loop in loops):
                continue
            if strict:
                labels_for_loops = []
                for loop in loops:
                    original = np.asarray(kept_vertices[np.asarray(loop, dtype=np.int64)],
                                          dtype=np.int64)
                    edge_votes = []
                    for u, v in zip(original, np.roll(original, -1)):
                        adjacent = edge_labels.get(tuple(sorted((int(u), int(v)))), set())
                        removed_adjacent = adjacent & {a, b}
                        if len(removed_adjacent) != 1:
                            edge_votes = []
                            break
                        edge_votes.append(next(iter(removed_adjacent)))
                    labels_for_loops.append(
                        edge_votes[0] if edge_votes and len(set(edge_votes)) == 1 else None
                    )
            else:
                labels_for_loops = [loop_label(loop, kept_vertices, edge_labels, {a, b})
                                    for loop in loops]
            if None in labels_for_loops or len(set(labels_for_loops)) != 2:
                continue
            lengths = [perimeter(damaged_vertices, loop) for loop in loops]
            target_index = int(np.argmin(lengths))
            distractor_index = 1 - target_index
            if not lengths[target_index] < lengths[distractor_index] * (1.0 - 1e-8):
                continue
            target_label = int(labels_for_loops[target_index])
            distractor_label = int(labels_for_loops[distractor_index])
            cue = cue_for_label(vertices, faces, labels, target_label)
            sample_id = (f"coseg_external_{category}_{path.stem}_target"
                         f"{target_label}_distractor{distractor_label}")
            return {
                "sample_id": sample_id,
                "category": category,
                "source_model_id": source_id,
                "raw_mesh": str(path.resolve()),
                "raw_segmentation": str(seg_path.resolve()),
                "source_mesh_sha256": mesh_sha256,
                "target_label": target_label,
                "distractor_label": distractor_label,
                "target_loop_vertices": np.asarray(loops[target_index], dtype=np.int64).tolist(),
                "distractor_loop_vertices": np.asarray(loops[distractor_index], dtype=np.int64).tolist(),
                "target_perimeter": lengths[target_index],
                "distractor_perimeter": lengths[distractor_index],
                "cue": cue.tolist(),
                "vertices": damaged_vertices,
                "faces": damaged_faces,
            }
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expand", action="store_true",
                        help="build a separate large-set expansion with strict interfaces")
    parser.add_argument("--holdout", action="store_true",
                        help="build an independent strict holdout after all prior protocols")
    parser.add_argument("--cross-category", action="store_true",
                        help="new source-disjoint semantic-target plus surface-distractor test")
    parser.add_argument("--per-category", type=int, default=None,
                        help="number of eligible source meshes per expansion category")
    parser.add_argument("--output", type=Path, default=None,
                        help="output directory; defaults to data/coseg_external_expansion")
    parser.add_argument("--allow-short", action="store_true",
                        help="keep all eligible meshes when a category has fewer than requested")
    args = parser.parse_args()

    if args.cross_category:
        if args.expand or args.holdout:
            parser.error("--cross-category cannot be combined with --expand/--holdout")
        build_cross_category(args.output or ROOT / "data/coseg_cross_category", args.per_category or 20)
        return

    if args.expand and args.holdout:
        parser.error("--expand and --holdout are mutually exclusive")
    if args.expand:
        sources = dict(EXPANSION_SOURCES)
        out = ROOT / "data" / "coseg_external_expansion" if args.output is None else args.output
        per_category = 35 if args.per_category is None else args.per_category
        sources = {key: (root, per_category) for key, (root, _) in sources.items()}
        strict = True
        include_expansion = False
    elif args.holdout:
        sources = HOLDOUT_SOURCES
        out = ROOT / "data" / "coseg_external_holdout" if args.output is None else args.output
        strict = True
        include_expansion = True
    else:
        sources = SOURCES
        out = OUT
        strict = False
        include_expansion = False
    if out.exists():
        raise FileExistsError(f"refusing to overwrite fixed benchmark: {out}")
    used_ids = used_source_ids()
    used_hashes = used_source_hashes(include_expansion) if (args.expand or args.holdout) else set()
    cases_dir = out / "cases"
    cases_dir.mkdir(parents=True)
    rows = []
    category_counts = {}
    for category in sources:
        candidates = []
        source_root, target_count = sources[category]
        candidate_hashes = set(used_hashes)
        shape_paths = list((source_root / "shapes").rglob("*.off"))
        for path in sorted(shape_paths,
                           key=lambda p: (int(p.stem) if p.stem.isdigit() else 10**9, p.stem)):
            item = candidate_for_mesh(category, source_root, path, used_ids,
                                       candidate_hashes if (args.expand or args.holdout) else None,
                                       strict=strict)
            if item is not None:
                candidates.append(item)
                if args.expand or args.holdout:
                    candidate_hashes.add(item["source_mesh_sha256"])
        if len(candidates) < target_count and not args.allow_short:
            raise RuntimeError(f"only {len(candidates)} eligible {category} sources; need {target_count}")
        chosen = candidates[:target_count]
        category_counts[category] = len(chosen)
        for item in chosen:
            output = cases_dir / f"{item['sample_id']}.npz"
            np.savez_compressed(output, vertices=item.pop("vertices"), faces=item.pop("faces"),
                                cue=np.asarray(item["cue"], dtype=float))
            item["input_path"] = str(output.resolve())
            item["input_sha256"] = sha256(output)
            item["true_interface_verified"] = True
            rows.append(item)
            used_ids.add(item["source_model_id"])
            if args.expand or args.holdout:
                used_hashes.add(item["source_mesh_sha256"])
    protocol = {
        "schema": ("coseg_external_expansion" if args.expand else
                   "coseg_external_holdout" if args.holdout else
                   "coseg_external_benchmark"),
        "source": ("Official Shape COSEG Tele-aliens, Large-Vases, and Large-Chairs"
                    if args.expand else
                    "Official Shape COSEG Large-Vases unused tail (independent holdout)"
                    if args.holdout else
                    "Official Shape COSEG Candelabra, Four-legged, Goblets, and Guitars"),
        "source_url": "https://www.yunhaiwang.net/public_html/ssl/ssd.htm",
        "n_cases": len(rows),
        "category_counts": category_counts,
        "selection": ("First eligible source-distinct mesh in fixed category and numeric source order; "
                      "remove two labeled segments, retain exactly two boundary loops, and designate "
                      "the smaller perimeter as target. Selection fixed before inference."),
        "label_assignment": ("Every retained boundary edge must be adjacent to exactly one removed "
                             "segment label; cue from the removed target segment. Labels are withheld "
                             "from inference and opened only during scoring."),
        "strict_interface_check": strict,
        "excluded_source_ids": sorted(used_source_ids()),
        "cases": rows,
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"n_cases": len(rows), "category_counts": category_counts,
                      "source_ids": [r["source_model_id"] for r in rows]},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
