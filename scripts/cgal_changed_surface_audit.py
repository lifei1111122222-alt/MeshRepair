"""Complete audit of newly introduced intersections, without rescanning fixed pairs.

Every new intersection must involve a new patch face or a retained face whose
vertex coordinates changed. Query those faces against the WHOLE mesh using
CGAL's native AABB triangle-soup index, then compare before/after face-pair
identities. This covers remote retained surfaces and moved-collar collisions;
it is not a seam-local approximation. Fixed/fixed inherited pairs are omitted.
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
from scripts.cgal_local_hole_adapter import _imports, _validate_boundary


def _query(vertices, faces, queried_faces):
    _imports()
    from CGAL.CGAL_Kernel import Point_3, Triangle_3, Segment_3, intersection
    from CGAL.CGAL_AABB_tree import AABB_tree_Triangle_3_soup
    vv, ff = np.asarray(vertices, dtype=float), np.asarray(faces, dtype=np.int64)
    points = [Point_3(*map(float, p)) for p in vv]
    triangles, native_to_face, degenerate = [], [], []
    face_to_native = {}
    for i, f in enumerate(ff):
        triangle = Triangle_3(points[int(f[0])], points[int(f[1])], points[int(f[2])])
        if triangle.is_degenerate():
            degenerate.append(i)
        else:
            face_to_native[i] = len(triangles)
            triangles.append(triangle)
            native_to_face.append(i)
    tree = AABB_tree_Triangle_3_soup(triangles)
    seen, pairs = set(), set()
    for i in sorted(set(map(int, queried_faces))):
        if i not in face_to_native:
            continue
        native_i = face_to_native[i]
        matches = []
        tree.all_intersected_primitives(triangles[native_i], matches)
        for native_j in matches:
            j = native_to_face[native_j]
            if i == j:
                continue
            pair = tuple(sorted((i, j)))
            if pair in seen:
                continue
            seen.add(pair)
            shared = set(map(int, ff[i])).intersection(map(int, ff[j]))
            if shared:
                obj = intersection(triangles[native_i], triangles[native_j])
                if obj.empty():
                    continue
                trivial = False
                if len(shared) == 1 and obj.is_Point_3():
                    trivial = obj.get_Point_3() == points[next(iter(shared))]
                elif len(shared) == 2:
                    a,b = sorted(shared)
                    edge = Segment_3(points[a], points[b])
                    if obj.is_Point_3():
                        trivial = edge.has_on(obj.get_Point_3())
                    elif obj.is_Segment_3():
                        seg = obj.get_Segment_3()
                        trivial = edge.has_on(seg.source()) and edge.has_on(seg.target())
                if trivial:
                    continue
            pairs.add(pair)
    return pairs, degenerate


def audit_changed_surface(before_vertices, retained_faces, after_vertices, patch_faces, selected_loop=None):
    bv, rf = np.asarray(before_vertices, float), np.asarray(retained_faces, np.int64)
    av, pf = np.asarray(after_vertices, float), np.asarray(patch_faces, np.int64).reshape(-1, 3)
    if len(av) < len(bv) or not np.isfinite(bv).all() or not np.isfinite(av).all():
        raise ValueError("Vertices are not a finite retained-prefix correspondence")
    # Exact coordinate comparison is intentional: any changed original face
    # can introduce an intersection, however small the optimizer movement.
    moved = np.any(bv != av[:len(bv)], axis=1)
    affected = np.flatnonzero(np.any(moved[rf], axis=1))
    new_ids = np.arange(len(rf), len(rf) + len(pf))
    before_pairs, before_degenerate = _query(bv, rf, affected)
    ff = np.concatenate([rf, pf])
    after_pairs, after_degenerate = _query(av, ff, np.concatenate([affected, new_ids]))
    introduced = after_pairs - before_pairs
    resolved = before_pairs - after_pairs
    group_counts = Counter()
    for a,b in introduced:
        group_counts["patch_internal" if a >= len(rf) and b >= len(rf)
                     else "patch_retained" if a >= len(rf) or b >= len(rf)
                     else "moved_retained_retained"] += 1
    has_degenerate = bool(before_degenerate or after_degenerate)
    result = {
        "status": "success", "algorithm": "CGAL AABB triangle soup + native triangle intersection",
        "scope": "all possible introduced pairs among nondegenerate faces; changed surfaces queried against the entire mesh",
        "retained_faces": len(rf), "patch_faces": len(pf),
        "moved_retained_vertices": int(np.sum(moved)), "affected_retained_faces": len(affected),
        "before_affected_pairs": [list(p) for p in sorted(before_pairs)],
        "after_affected_pairs": [list(p) for p in sorted(after_pairs)],
        "introduced_pairs": [list(p) for p in sorted(introduced)],
        "resolved_pairs": [list(p) for p in sorted(resolved)],
        "introduced_pair_count": len(introduced), "resolved_pair_count": len(resolved),
        "introduced_groups": dict(group_counts),
        "introduced_intersection_nondegenerate": bool(introduced),
        "introduced_intersection": True if introduced else (None if has_degenerate else False),
        "before_degenerate_faces": before_degenerate, "after_degenerate_faces": after_degenerate,
        "degenerate_caveat": "Degenerate faces are outside native AABB support: negative full-mesh claims remain unknown if any exist",
    }
    if selected_loop is not None:
        loop = np.asarray(selected_loop, np.int64)
        valid_cycle = True
        try:
            _validate_boundary(bv, rf, loop)
        except ValueError as exc:
            valid_cycle = False
            result["input_cycle_error"] = str(exc)
        def edgecounts(fs):
            return Counter(tuple(sorted((int(a),int(b)))) for f in fs for a,b in zip(f,np.roll(f,-1)))
        patch_counts, output_counts = edgecounts(pf), edgecounts(ff)
        target_edges = {tuple(sorted((int(a),int(b)))) for a,b in zip(loop,np.roll(loop,-1))}
        result["input_loop_is_simple_boundary_cycle"] = valid_cycle
        result["strict_selected_loop_closure"] = bool(valid_cycle and len(pf) and
            all(output_counts[e] == 2 and patch_counts[e] == 1 for e in target_edges) and
            {e for e,n in patch_counts.items() if n == 1} == target_edges)
    return result


def worker(args):
    try:
        with np.load(args.input, allow_pickle=False) as data:
            bv, rf, loop = data["vertices"], data["faces"], data["selected_loop"]
        with np.load(args.patch, allow_pickle=False) as p:
            av, pf = p["vertices"], p["patch_faces"]
        from tcmr.validity import orient_patch_to_boundary
        pf = orient_patch_to_boundary(pf, rf, loop)
        result = audit_changed_surface(bv, rf, av, pf, loop)
    except Exception as exc:
        result = {"status": "unavailable", "error": str(exc), "error_type": type(exc).__name__}
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")


def audit_pilot(args):
    pilot_path = Path(args.pilot)
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    tcmr = json.loads(Path(args.tcmr).read_text(encoding="utf-8")) if args.tcmr else None
    methods = list(pilot["methods"])
    if tcmr:
        if [r["sample_id"] for r in tcmr["rows"]] != pilot["provenance"]["sample_ids"]:
            raise ValueError("TCMR sample order mismatch")
        methods.append("tcmr")
    outpath = Path(args.output)
    if outpath.exists():
        raise FileExistsError(outpath)
    run_dir = outpath.with_suffix("")
    run_dir.mkdir(exist_ok=True)
    rows = []
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    provenance = {"pilot_sha256": hashlib.sha256(pilot_path.read_bytes()).hexdigest(),
                  "tcmr_sha256": hashlib.sha256(Path(args.tcmr).read_bytes()).hexdigest() if args.tcmr else None,
                  "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "metric": "complete possible-new-pair audit; inherited unchanged/unchanged pairs not counted"}
    for i,sid in enumerate(pilot["provenance"]["sample_ids"]):
        inp = pilot_path.with_suffix("") / f"{i:05d}_damaged_only.npz"
        for method in methods:
            methodrows = tcmr["rows"] if method == "tcmr" else pilot["methods"][method]["rows"]
            if methodrows[i]["status"] != "success":
                rows.append({"sample_id": sid, "method": method, "status": "no_returned_patch", "strict_selected_loop_closure": False})
                continue
            patch_dir = Path(args.tcmr).with_suffix("") if method == "tcmr" else pilot_path.with_suffix("")
            patch = patch_dir / f"{i:05d}_{method}.npz"
            dest = run_dir / f"{i:05d}_{method}.json"
            t0 = time.perf_counter()
            try:
                proc = subprocess.run([sys.executable,__file__,"--input",str(inp),"--patch",str(patch),"--output",str(dest)],
                                      cwd=ROOT,env=env,capture_output=True,text=True,timeout=args.timeout)
                (run_dir/f"{i:05d}_{method}.stderr.txt").write_text(proc.stderr,encoding="utf-8")
                result = json.loads(dest.read_text(encoding="utf-8")) if proc.returncode == 0 and dest.exists() else {
                    "status": "native_crash", "returncode": proc.returncode}
            except subprocess.TimeoutExpired:
                result = {"status": "timeout"}
            row = {"sample_id":sid,"method":method,"runtime_seconds":time.perf_counter()-t0,**result}
            rows.append(row)
            print(json.dumps({k:v for k,v in row.items() if not isinstance(v,list)}),flush=True)
            outpath.write_text(json.dumps({"complete":False,"provenance":provenance,"rows":rows},indent=2),encoding="utf-8")
    summary = {}
    for method in methods:
        rs = [r for r in rows if r["method"] == method]
        measured = [r for r in rs if r["status"] == "success"]
        summary[method] = {"n":len(rs),"status_counts":dict(Counter(r["status"] for r in rs)),
            "measured_n":len(measured),"introduced_nondegenerate_n":sum(r["introduced_intersection_nondegenerate"] for r in measured),
            "introduced_unknown_n":sum(r["introduced_intersection"] is None for r in measured),
            "strict_selected_closure_n":sum(r.get("strict_selected_loop_closure",False) for r in rs)}
    outpath.write_text(json.dumps({"complete":True,"provenance":provenance,"summary":summary,"rows":rows},indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input");p.add_argument("--patch");p.add_argument("--pilot");p.add_argument("--tcmr")
    p.add_argument("--output",required=True);p.add_argument("--timeout",type=float,default=120)
    a=p.parse_args()
    audit_pilot(a) if a.pilot else worker(a)


if __name__ == "__main__":
    main()
