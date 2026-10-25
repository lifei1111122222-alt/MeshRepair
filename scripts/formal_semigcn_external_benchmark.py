"""Run the official SeMIGCN/MGCN refinement on the fixed external protocol.

This is an explicitly adapted learning baseline: target-loop selection uses the
same cue-nearest rule as the CGAL comparator, then the official no-GT MGCN
refinement is applied to a shared CGAL initialization.  No target labels are
used before scoring.  The adaptation is recorded so it is not presented as a
full reproduction of the authors' global preprocessing pipeline.
"""
from __future__ import annotations

import collections
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BENCH = ROOT / "experiments" / "formal_coseg_external_benchmark"
OUT = BENCH / "semigcn"
READINESS = ROOT / "experiments" / "semigcn_readiness_20261008_v1"
AUTHOR = ROOT / "experiments" / "semigcn_cpu_author_v1"
SEMIGCN_PYTHON = ROOT / ".external_baselines" / "semigcn_env" / "Scripts" / "python.exe"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for vertex in vertices:
            fh.write("v " + " ".join(format(float(x), ".12g") for x in vertex) + "\n")
        for face in faces:
            fh.write("f " + " ".join(str(int(x) + 1) for x in face) + "\n")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def cue_nearest_selection(prediction: dict, case: dict) -> tuple[tuple, list[tuple]] | None:
    cue = np.asarray(case["cue"][:3], dtype=float)
    options = []
    for loop in prediction.get("loops", []):
        cgal = next((c for c in loop.get("candidates", [])
                     if c.get("method") == "cgal_refine_fair"), None)
        if cgal is None:
            continue
        with np.load(loop["input_path"], allow_pickle=False) as z:
            vertices = np.asarray(z["vertices"], dtype=float)
            selected = np.asarray(z["selected_loop"], dtype=np.int64)
        distance = float(np.linalg.norm(vertices[selected].mean(axis=0) - cue))
        key = (distance, int(loop["loop_id"]))
        options.append((key, loop, cgal, Path(cgal["patch_path"]), selected))
    if not options:
        return None
    options.sort(key=lambda x: x[0])
    return options[0], options


def prepare_case(case: dict, prediction: dict, output_root: Path | None = None) -> dict | None:
    selection = cue_nearest_selection(prediction, case)
    if selection is None:
        return None
    selected, all_options = selection
    _, loop_row, cgal, patch_path, selected_loop = selected
    case_root = (output_root or OUT) / "cases" / f"case{case['index']:03d}"
    if case_root.exists():
        return json.loads((case_root / "prepared_manifest.json").read_text(encoding="utf-8"))
    case_root.mkdir(parents=True)
    with np.load(case["input_path"], allow_pickle=False) as z:
        vin = np.asarray(z["vertices"], dtype=float)
        fin = np.asarray(z["faces"], dtype=np.int64)
    initializer_vertices = np.asarray(vin, dtype=float).copy()
    patch_faces_all = []
    filled_loop_ids = []
    for _, loop, _candidate, patch_path, _ in all_options:
        with np.load(patch_path, allow_pickle=False) as z:
            patch_vertices = np.asarray(z["vertices"], dtype=float)
            patch_faces = np.asarray(z["patch_faces"], dtype=np.int64)
        if len(patch_vertices) < len(vin) or not np.array_equal(patch_vertices[:len(vin)], vin):
            raise RuntimeError(f"initializer does not preserve retained vertices: {case['sample_id']}")
        offset = len(initializer_vertices) - len(vin)
        initializer_vertices = np.vstack([initializer_vertices, patch_vertices[len(vin):]])
        adjusted_faces = patch_faces.copy()
        inserted = adjusted_faces >= len(vin)
        adjusted_faces[inserted] += offset
        patch_faces_all.append(adjusted_faces)
        filled_loop_ids.append(int(loop["loop_id"]))
    patch_faces = np.vstack(patch_faces_all)
    retained_face_ids = np.arange(len(fin), dtype=np.int64)
    global_ids = np.arange(len(initializer_vertices), dtype=np.int64)
    inverse = global_ids.copy()
    local_faces = np.vstack([fin, patch_faces])
    local_vertices = initializer_vertices
    observed = global_ids < len(vin)
    edges = np.sort(np.concatenate([local_faces[:, [0, 1]], local_faces[:, [1, 2]],
                                    local_faces[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    if not np.all(counts == 2):
        raise RuntimeError(f"common initializer is not closed: {case['sample_id']}")
    lengths = np.linalg.norm(local_vertices[edges[:, 0]] - local_vertices[edges[:, 1]], axis=1)
    scale = 1 / float(max(lengths.mean(), 1e-8))
    center = (local_vertices.max(axis=0) + local_vertices.min(axis=0)) / 2
    normalized = (local_vertices - center) * scale
    name = case["sample_id"]
    run_name = case_root.name
    write_obj(case_root / f"{run_name}_initial.obj", normalized, local_faces)
    write_obj(case_root / f"{run_name}_original.obj", normalized[observed], inverse[fin])
    (case_root / f"{run_name}_vmask.json").write_text(json.dumps(observed.tolist()), encoding="utf-8")
    np.savez(case_root / "restore_mapping.npz", global_ids=global_ids,
             retained_face_ids=retained_face_ids, vertices=vin, faces=fin,
             patch_faces=patch_faces, initializer_vertices=initializer_vertices,
             selected_loop=selected_loop)
    transform = {"scale": scale, "offset": (-center * scale).tolist()}
    dump(case_root / "normalization_transform.json", transform)
    manifest = {
        "sample_id": case["sample_id"], "index": case["index"],
        "input_path": case["input_path"], "input_sha256": case["input_sha256"],
        "selected_loop_id": int(loop_row["loop_id"]),
        "filled_loop_ids": filled_loop_ids,
        "selection_rule": "cue-nearest among valid boundary loops; no target labels",
        "track": "adapted cue-nearest common-CGAL-initialized SeMIGCN/MGCN",
        "iterations": 20, "gt_geometry_read": False, "target_labels_read": False,
        "files": {p.name: sha(p) for p in case_root.iterdir() if p.is_file()},
    }
    dump(case_root / "manifest.json", manifest)
    manifest["folder"] = str(case_root.resolve())
    dump(case_root / "prepared_manifest.json", manifest)
    return manifest


def prepare_raw_case(case: dict, output_root: Path) -> dict:
    """Rerun the unchanged initializer from damaged input, with fresh timers."""
    from tcmr.boundary import extract_boundary_loops
    from tcmr.validity import orient_patch_to_boundary
    from scripts.local_repair_topology import inspect_target_cycle
    from scripts.cgal_local_hole_adapter import repair

    started = time.perf_counter()
    if sha(Path(case["input_path"])) != case["input_sha256"]:
        raise RuntimeError("Damaged input differs from the frozen protocol")
    with np.load(case["input_path"], allow_pickle=False) as z:
        vertices = np.asarray(z["vertices"], float)
        faces = np.asarray(z["faces"], np.int64)
        cue = np.asarray(z["cue"], float)
    root = output_root / "initializer" / f"case{case['index']:03d}"
    root.mkdir(parents=True, exist_ok=False)
    loops = extract_boundary_loops(vertices, faces)
    rows = []
    repair_seconds = 0.
    for loop_id, loop in enumerate(loops):
        if not inspect_target_cycle(faces, loop)["input_simple_closed_boundary"]:
            continue
        loop = np.asarray(loop, np.int64)
        damaged_path = root / f"loop{loop_id}_input.npz"
        patch_path = root / f"loop{loop_id}_cgal.npz"
        np.savez_compressed(damaged_path, vertices=vertices, faces=faces, selected_loop=loop)
        tick = time.perf_counter()
        after, patch, native = repair(vertices, faces, loop, "cgal_refine_fair")
        patch = orient_patch_to_boundary(patch, faces, loop)
        repair_seconds += time.perf_counter() - tick
        np.savez_compressed(patch_path, vertices=after, patch_faces=patch)
        rows.append({"loop_id": loop_id, "input_path": str(damaged_path.resolve()),
                     "candidates": [{"method": "cgal_refine_fair", "patch_path": str(patch_path.resolve()),
                                     "native_details": native}]})
    sanitized = {k: case[k] for k in ("sample_id", "index", "input_path", "input_sha256")}
    sanitized["cue"] = cue.tolist()
    manifest = prepare_case(sanitized, {"loops": rows}, output_root)
    if manifest is None:
        raise RuntimeError("No valid input boundary for SeMIGCN initialization")
    manifest["fresh_preparation_timing"] = {
        "raw_input_to_prepared_seconds": time.perf_counter() - started,
        "cgal_geometry_seconds": repair_seconds, "source": "fresh damaged input",
        "target_labels_read": False,
        "scope": "input read, valid boundary enumeration, cue selection, all-loop CGAL initialization, normalization and input export"}
    dump(Path(manifest["folder"]) / "prepared_manifest.json", manifest)
    return manifest


def prepare(case_index: int | None = None, fresh_initializer: bool = False, iterations: int = 20) -> None:
    protocol = json.loads((BENCH / "protocol.json").read_text(encoding="utf-8"))
    predictions = {p["sample_id"]: p for p in
                  (json.loads(path.read_text(encoding="utf-8"))
                   for path in sorted((BENCH / "predictions").glob("*.json")))} if not fresh_initializer else {}
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in protocol["cases"]:
        if case_index is not None and int(case["index"]) != case_index:
            continue
        try:
            manifest = (prepare_raw_case(case, OUT) if fresh_initializer else
                        prepare_case(case, predictions[case["sample_id"]]))
            if manifest is None:
                raise RuntimeError("No cue-selectable valid CGAL initialization")
        except Exception as exc:
            # A fresh timed single-case preparation must fail its worker.
            # Batch preparation keeps every frozen source in the denominator.
            if fresh_initializer:
                raise
            failure_path = OUT / "cases" / f"case{case['index']:03d}" / "preparation_failed.json"
            manifest = {k: case[k] for k in ("sample_id", "index", "input_path", "input_sha256")}
            manifest.update({"status": "failed", "reason": "preparation_failure", "error": repr(exc),
                             "preparation_failure_path": str(failure_path.resolve()),
                             "target_labels_read": False})
            dump(failure_path, manifest)
            rows.append(manifest)
            print(json.dumps({"preparation_failed": case["index"], "sample_id": case["sample_id"],
                              "error": repr(exc)}, ensure_ascii=False), flush=True)
            continue
        if manifest is not None and manifest.get("iterations") != iterations:
            manifest["iterations"] = iterations
            dump(Path(manifest["folder"]) / "prepared_manifest.json", manifest)
            summary = dict(manifest)
            summary.pop("folder", None)
            dump(Path(manifest["folder"]) / "manifest.json", summary)
        rows.append(manifest)
    dump(OUT / "protocol.json", {"schema": "formal_semigcn_external_benchmark",
                                  "n_cases": len(rows), "iterations": iterations,
                                  "adaptation": "cue-nearest selection plus shared CGAL initialization; official MGCN refinement; no target labels before scoring",
                                  "cases": rows})
    print(json.dumps({"prepared": sum(r.get("status") != "failed" for r in rows),
                      "failed": sum(r.get("status") == "failed" for r in rows),
                      "n": len(rows)}, ensure_ascii=False))


def run() -> None:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    status_path = OUT / "run_status.json"
    rows = json.loads(status_path.read_text(encoding="utf-8")).get("rows", []) if status_path.exists() else []
    done = {row["sample_id"] for row in rows if row.get("status") == "completed"}
    environment = dict(os.environ, PYTHONHASHSEED="314", OMP_NUM_THREADS="2",
                       OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2",
                       NUMEXPR_NUM_THREADS="2", SEMIGCN_THREADS="2",
                       PYTHONUNBUFFERED="1")
    pending = [case for case in protocol["cases"] if case["sample_id"] not in done]
    workers = max(1, int(os.environ.get("SEMIGCN_WORKERS", "3")))

    def one(case):
        root = OUT / "cases" / f"case{case['index']:03d}"
        started = time.perf_counter()
        row = {"sample_id": case["sample_id"], "index": case["index"], "status": "running"}
        code = ("import importlib.util,sys; from pathlib import Path; "
                f"s=importlib.util.spec_from_file_location('stage',r'{(READINESS / 'run_stage.py').as_posix()}'); "
                "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
                f"m.ROOT=Path(r'{root.resolve()}'); m.AUTHOR=Path(r'{AUTHOR.resolve()}'); "
                "sys.argv=['run_stage.py','--iterations','20','--stage','external']; m.main()")
        try:
            log_path = root / "external.log"
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run([str(SEMIGCN_PYTHON), "-c", code], cwd=ROOT,
                                            env=environment, stdout=log, stderr=subprocess.STDOUT,
                                            check=False)
            row["status"] = "completed" if completed.returncode == 0 else "failed"
            row["returncode"] = int(completed.returncode)
        except Exception as exc:  # pragma: no cover - external baseline failure path
            row["status"] = "failed"
            row["error"] = repr(exc)
        row["elapsed_seconds"] = time.perf_counter() - started
        return row

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(one, case) for case in pending]
        for index, future in enumerate(as_completed(futures), 1):
            row = future.result()
            rows.append(row)
            dump(status_path, {"complete": False, "rows": rows})
            print(json.dumps({"run": index, "pending": len(pending), **row}, ensure_ascii=False), flush=True)
    dump(status_path, {"complete": True, "rows": rows})


def score() -> None:
    from tcmr.validity import check_mesh_validity

    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    labels = {row["sample_id"]: row for row in
              json.loads((BENCH / "protocol.json").read_text(encoding="utf-8"))["cases"]}
    run_status = {row["sample_id"]: row for row in
                  json.loads((OUT / "run_status.json").read_text(encoding="utf-8"))["rows"]}
    rows = []
    available = {row["sample_id"] for row in run_status.values()
                 if row.get("status") == "completed"}
    for case in protocol["cases"]:
        if case["sample_id"] not in available:
            continue
        root = OUT / "cases" / f"case{case['index']:03d}"
        status = run_status.get(case["sample_id"], {})
        mapping = np.load(root / "restore_mapping.npz", allow_pickle=False)
        vin, fin = mapping["vertices"], mapping["faces"]
        selected = mapping["selected_loop"]
        destination = root / "external" / root.name
        patch_file = destination / "restored_patch.npz"
        target_loop = labels[case["sample_id"]]["target_loop_vertices"]
        result = {"sample_id": case["sample_id"], "category": labels[case["sample_id"]].get("category"),
                  "status": status.get("status"), "selected_correct": bool(set(selected.tolist()) == set(target_loop))}
        if status.get("status") != "completed" or not patch_file.exists():
            result.update({"correct_safe": False, "error": status.get("error", "missing output")})
        else:
            with np.load(patch_file, allow_pickle=False) as z:
                vertices = np.asarray(z["vertices"], dtype=float)
                patch_faces = np.asarray(z["patch_faces"], dtype=np.int64)
            faces = np.vstack([fin, patch_faces])
            validity = check_mesh_validity(vertices, faces, selected)
            result.update({"correct_safe": bool(result["selected_correct"] and
                                                  validity.get("self_intersections") in (0, None) and
                                                  validity.get("degenerate_faces", 1) == 0),
                           "validity": validity,
                           "retained_displacement_mean": float(np.mean(np.linalg.norm(vertices[:len(vin)] - vin, axis=1)))})
        rows.append(result)
    successful = [row for row in rows if row.get("status") == "completed"]
    report = {"schema": "formal_semigcn_external_benchmark_report", "complete": True,
              "n": len(rows), "protocol_n": len(protocol["cases"]),
              "resource_limited_prefix": True, "successful": len(successful),
              "correct_safe_n": sum(bool(row.get("correct_safe")) for row in rows),
              "selected_correct_n": sum(bool(row.get("selected_correct")) for row in rows),
              "correct_safe_rate": (sum(bool(row.get("correct_safe")) for row in rows) / len(rows)) if rows else None,
              "selected_correct_rate": (sum(bool(row.get("selected_correct")) for row in rows) / len(rows)) if rows else None,
              "correct_safe_rate_ci95": ([float(x) for x in binomtest(sum(bool(row.get("correct_safe")) for row in rows), len(rows)).proportion_ci(confidence_level=0.95, method="exact")]
                                          if rows else [None, None]),
              "selected_correct_rate_ci95": ([float(x) for x in binomtest(sum(bool(row.get("selected_correct")) for row in rows), len(rows)).proportion_ci(confidence_level=0.95, method="exact")]
                                              if rows else [None, None]),
              "rows": rows}
    dump(OUT / "report.json", report)
    print(json.dumps({k: report[k] for k in ("n", "successful", "selected_correct_n", "correct_safe_n")}, ensure_ascii=False))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "run", "score"))
    parser.add_argument("--benchmark-root", type=Path, default=BENCH)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--case-index", type=int)
    parser.add_argument("--iterations", type=int, default=20, help="declared fitting budget; preparation does not train")
    parser.add_argument("--fresh-initializer", action="store_true",
                        help="Rerun CGAL from damaged input and measure preparation; use a fresh output root")
    args = parser.parse_args()
    BENCH = args.benchmark_root.resolve()
    OUT = args.output_root.resolve() if args.output_root else BENCH / "semigcn"
    if args.stage == "prepare":
        prepare(args.case_index, args.fresh_initializer, args.iterations)
    else:
        {"run": run, "score": score}[args.stage]()
