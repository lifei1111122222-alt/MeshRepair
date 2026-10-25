"""Topology-independent, whole-mesh intersection audit using official CGAL.

CGAL AABB triangle-soup queries supply broad/narrow phase intersection pairs.
For faces sharing vertex IDs, only the expected shared point/edge contact is
excluded. Shared-vertex area overlap is therefore retained. Degenerate faces
are reported separately because CGAL AABB predicates require nondegeneracy.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.cgal_local_hole_adapter import _imports


def intersections(vertices, faces, retained_face_count):
    _imports()
    from CGAL.CGAL_Kernel import Point_3, Triangle_3, Segment_3, intersection
    from CGAL.CGAL_AABB_tree import AABB_tree_Triangle_3_soup
    vv, ff = np.asarray(vertices, dtype=float), np.asarray(faces, dtype=np.int64)
    if not np.isfinite(vv).all():
        raise ValueError("Nonfinite mesh coordinates")
    points = [Point_3(*map(float, p)) for p in vv]
    facesets = [set(map(int, f)) for f in ff]
    triangles, native_to_face, degenerate = [], [], []
    for i, f in enumerate(ff):
        triangle = Triangle_3(points[int(f[0])], points[int(f[1])], points[int(f[2])])
        if triangle.is_degenerate():
            degenerate.append(i)
        else:
            triangles.append(triangle)
            native_to_face.append(i)
    tree = AABB_tree_Triangle_3_soup(triangles)
    pairs, expected_contacts = [], 0
    for native_i, triangle in enumerate(triangles):
        matches = []
        tree.all_intersected_primitives(triangle, matches)
        i = native_to_face[native_i]
        for native_j in matches:
            if native_j <= native_i:
                continue
            j = native_to_face[native_j]
            shared = facesets[i].intersection(facesets[j])
            if shared:
                obj = intersection(triangle, triangles[native_j])
                if obj.empty():
                    continue
                trivial = False
                if len(shared) == 1 and obj.is_Point_3():
                    trivial = obj.get_Point_3() == points[next(iter(shared))]
                elif len(shared) == 2:
                    a, b = sorted(shared)
                    edge = Segment_3(points[a], points[b])
                    if obj.is_Point_3():
                        trivial = edge.has_on(obj.get_Point_3())
                    elif obj.is_Segment_3():
                        segment = obj.get_Segment_3()
                        trivial = edge.has_on(segment.source()) and edge.has_on(segment.target())
                if trivial:
                    expected_contacts += 1
                    continue
            pairs.append([i, j])
    pairs.sort()
    groups = Counter()
    shared_nontrivial = 0
    for i, j in pairs:
        pi, pj = i >= retained_face_count, j >= retained_face_count
        groups["patch_internal" if pi and pj else "patch_retained" if pi or pj else "retained_retained"] += 1
        shared_nontrivial += bool(facesets[i].intersection(facesets[j]))
    return {
        "status": "success", "algorithm": "CGAL AABB_tree_Triangle_3_soup + CGAL::intersection(Triangle_3, Triangle_3)",
        "scope": "all nondegenerate face pairs; only topologically expected shared-point/shared-edge contact excluded",
        "face_count": len(ff), "evaluated_nondegenerate_face_count": len(triangles),
        "degenerate_face_count": len(degenerate), "degenerate_face_indices": degenerate,
        "degenerate_policy": "Excluded from AABB queries and reported separately; intersection count is restricted to nondegenerate faces",
        "count": len(pairs), "groups": dict(groups), "pair_indices": pairs,
        "shared_vertex_nontrivial_count": shared_nontrivial,
        "expected_shared_contacts_excluded": expected_contacts,
        "new_patch_has_intersection": bool(groups["patch_internal"] + groups["patch_retained"]),
    }


def worker(args):
    try:
        with np.load(args.input, allow_pickle=False) as data:
            vv, ff = data["vertices"], data["faces"]
            loop = data["selected_loop"] if "selected_loop" in data else None
        retained = len(ff)
        if args.patch:
            with np.load(args.patch, allow_pickle=False) as patch:
                vv, pf = patch["vertices"], patch["patch_faces"]
            if loop is not None:
                from tcmr.validity import orient_patch_to_boundary
                pf = orient_patch_to_boundary(pf, ff, loop)
            ff = np.concatenate([ff, pf])
        result = intersections(vv, ff, retained)
    except Exception as exc:
        result = {"status": "unavailable", "error": str(exc), "error_type": type(exc).__name__}
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k not in ("pair_indices", "degenerate_face_indices")}), flush=True)


def audit_pilot(args):
    source = Path(args.pilot)
    pilot = json.loads(source.read_text(encoding="utf-8"))
    pilot_dir = source.with_suffix("")
    output = Path(args.output)
    run_dir = output.with_suffix("")
    if output.exists():
        raise FileExistsError(output)
    run_dir.mkdir(exist_ok=True)
    methods = list(pilot["methods"])
    tcmr = json.loads(Path(args.tcmr).read_text(encoding="utf-8")) if args.tcmr else None
    if tcmr:
        if [r["sample_id"] for r in tcmr["rows"]] != pilot["provenance"]["sample_ids"]:
            raise ValueError("TCMR sample set/order differs")
        methods.append("tcmr")
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    rows = []
    for i,sid in enumerate(pilot["provenance"]["sample_ids"]):
        inp = pilot_dir / f"{i:05d}_damaged_only.npz"
        before_pairs = None
        for method in ["before"] + methods:
            if method != "before":
                result_rows = tcmr["rows"] if method == "tcmr" else pilot["methods"][method]["rows"]
                if result_rows[i]["status"] != "success":
                    rows.append({"sample_id": sid, "method": method, "status": "no_returned_patch"})
                    continue
            out = run_dir / f"{i:05d}_{method}.json"
            cmd = [sys.executable, __file__, "--input", str(inp), "--output", str(out)]
            if method != "before":
                directory = Path(args.tcmr).with_suffix("") if method == "tcmr" else pilot_dir
                cmd += ["--patch", str(directory / f"{i:05d}_{method}.npz")]
            t0 = time.perf_counter()
            try:
                proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=args.timeout)
                (run_dir / f"{i:05d}_{method}.stderr.txt").write_text(proc.stderr, encoding="utf-8")
                result = json.loads(out.read_text(encoding="utf-8")) if proc.returncode == 0 and out.exists() else {
                    "status": "native_crash", "returncode": proc.returncode}
            except subprocess.TimeoutExpired:
                result = {"status": "timeout"}
            pairset = {tuple(pair) for pair in result.get("pair_indices", [])}
            if result["status"] == "success":
                if method == "before":
                    before_pairs = pairset
                elif before_pairs is not None:
                    result["introduced_pair_count"] = len(pairset - before_pairs)
                    result["resolved_pair_count"] = len(before_pairs - pairset)
                    result["introduced_intersection"] = bool(pairset - before_pairs)
            row = {"sample_id": sid, "method": method, "runtime_seconds": time.perf_counter()-t0, **result}
            rows.append(row)
            print(json.dumps({k:v for k,v in row.items() if k not in ("pair_indices", "degenerate_face_indices")}), flush=True)
            output.write_text(json.dumps({"complete": False, "rows": rows}, indent=2), encoding="utf-8")
    summaries = {}
    for method in ["before"] + methods:
        rs = [r for r in rows if r["method"] == method]
        eligible = [r for r in rs if r["status"] == "success"]
        summaries[method] = {"n": len(rs), "status_counts": dict(Counter(r["status"] for r in rs)),
                             "measured_n": len(eligible), "introduced_intersection_n": sum(r.get("introduced_intersection",False) for r in eligible),
                             "patch_has_intersection_n": sum(r["new_patch_has_intersection"] for r in eligible),
                             "meshes_with_degenerate_faces_n": sum(r["degenerate_face_count"] > 0 for r in eligible)}
    payload = {"complete": True, "source_pilot": str(source.resolve()),
               "source_pilot_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
               "source_tcmr": args.tcmr, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "summary": summaries, "rows": rows}
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    parser.add_argument("--patch")
    parser.add_argument("--pilot")
    parser.add_argument("--tcmr")
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    audit_pilot(args) if args.pilot else worker(args)


if __name__ == "__main__":
    main()
