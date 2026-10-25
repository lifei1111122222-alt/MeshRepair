"""Run the pinned SeMIGCN/MGCN baseline, including budget and seed protocols.

The bundled author source is unmodified. Use the configured Python environment
with the author dependencies; the author model automatically selects CUDA when
available. SEMIGCN_SCORING_PYTHON can select a separate evaluation interpreter.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
BENCH_ROOT = REPO / "experiments" / "formal_coseg_external_benchmark"
HERE = BENCH_ROOT / "semigcn_gpu"
CPU_ROOT = BENCH_ROOT / "semigcn"
AUTHOR = REPO / "baselines" / "semigcn"
STAGE_RUNNER = REPO / "scripts" / "semigcn_stage.py"
SCORING_PYTHON = Path(os.environ.get("SEMIGCN_SCORING_PYTHON", sys.executable))


def load_stage():
    spec = importlib.util.spec_from_file_location("semigcn_gpu_stage", STAGE_RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def verify_fresh_preparation(expected: dict, actual: dict) -> dict:
    """Allow native floating-point roundoff, never a different mesh/selection."""
    import numpy as np
    for key in ("input_sha256", "selected_loop_id", "filled_loop_ids"):
        if expected[key] != actual[key]:
            raise RuntimeError(f"fresh common initialization differs in {key}")
    old_root, new_root = Path(expected["folder"]), Path(actual["folder"])
    with np.load(old_root / "restore_mapping.npz", allow_pickle=False) as a, np.load(new_root / "restore_mapping.npz", allow_pickle=False) as b:
        if set(a.files) != set(b.files):
            raise RuntimeError("fresh initializer mapping keys differ")
        edges = np.concatenate([a["faces"][:, [0, 1]], a["faces"][:, [1, 2]], a["faces"][:, [2, 0]]])
        edge_length = float(np.median(np.linalg.norm(a["vertices"][edges[:, 0]] - a["vertices"][edges[:, 1]], axis=1)))
        tolerance = max(edge_length * 1e-10, np.finfo(float).eps)
        maximum = 0.
        for key in a.files:
            if key == "initializer_vertices":
                if a[key].shape != b[key].shape or not np.allclose(a[key], b[key], rtol=0., atol=tolerance):
                    raise RuntimeError("fresh initializer coordinates differ beyond native roundoff")
                maximum = float(np.max(np.abs(a[key] - b[key])))
            elif not np.array_equal(a[key], b[key]):
                raise RuntimeError(f"fresh initializer discrete/observed mapping differs: {key}")
    for filename in expected["files"]:
        old, new = old_root / filename, new_root / filename
        if filename.endswith(".obj"):
            old_lines, new_lines = old.read_text().splitlines(), new.read_text().splitlines()
            if len(old_lines) != len(new_lines):
                raise RuntimeError(f"fresh initializer OBJ size differs: {filename}")
            for left, right in zip(old_lines, new_lines):
                a, b = left.split(), right.split()
                if a[0] != b[0] or (a[0] == "f" and a != b):
                    raise RuntimeError(f"fresh initializer OBJ faces differ: {filename}")
                if a[0] == "v" and not np.allclose(np.asarray(a[1:], float), np.asarray(b[1:], float), rtol=0., atol=1e-9):
                    raise RuntimeError(f"fresh normalized initializer vertices differ: {filename}")
        elif filename == "normalization_transform.json":
            a, b = json.loads(old.read_text()), json.loads(new.read_text())
            if not np.allclose([a["scale"], *a["offset"]], [b["scale"], *b["offset"]], rtol=1e-12, atol=1e-12):
                raise RuntimeError("fresh normalization differs")
        elif not filename.endswith(".npz") and old.read_bytes() != new.read_bytes():
            raise RuntimeError(f"fresh initializer declared input differs: {filename}")
    return {"retained_vertices_faces_and_selection_exact": True,
            "initializer_max_abs_difference": maximum, "initializer_absolute_tolerance": tolerance,
            "tolerance_rule": "1e-10 times input median edge length; discrete topology and observed vertices exact"}


def run(case: str, iterations: int, stage_name: str, baseline_stage: str | None = None,
        seed: int = 314, remeasure_preparation: bool = False) -> None:
    execution_started = time.perf_counter()
    source = CPU_ROOT / "cases" / case
    if not (source / "prepared_manifest.json").exists():
        raise FileNotFoundError(f"prepared case missing: {source}")
    output_root = HERE / "cases" / case
    if output_root.exists():
        if not (output_root / "prepared_manifest.json").exists():
            raise FileExistsError(f"existing GPU case lacks prepared manifest: {output_root}")
        if (output_root / stage_name).exists():
            raise FileExistsError(f"refusing to overwrite existing GPU stage: {output_root / stage_name}")
    else:
        output_root.mkdir(parents=True)
        shutil.copy2(source / "prepared_manifest.json", output_root / "prepared_manifest.json")
    preparation_wall_seconds = None
    if remeasure_preparation:
        # This fresh CPU stage is included in measured wall time. The earlier
        # prepared inputs provide an exact adapter-equivalence check only.
        fresh_root = output_root / "runtime_preparation"
        started = time.perf_counter()
        command = [str(SCORING_PYTHON), str(REPO / "scripts/formal_semigcn_external_benchmark.py"),
                   "prepare", "--benchmark-root", str(BENCH_ROOT), "--output-root", str(fresh_root),
                   "--case-index", str(int(case.removeprefix("case"))), "--iterations", str(iterations), "--fresh-initializer"]
        subprocess.run(command, cwd=REPO, check=True)
        preparation_wall_seconds = time.perf_counter() - started
        actual_manifest = fresh_root / "cases" / case / "prepared_manifest.json"
        expected = json.loads((source / "prepared_manifest.json").read_text(encoding="utf-8"))
        actual = json.loads(actual_manifest.read_text(encoding="utf-8"))
        equivalence_started = time.perf_counter()
        equivalence = verify_fresh_preparation(expected, actual)
        equivalence_seconds = time.perf_counter() - equivalence_started
        shutil.copy2(actual_manifest, output_root / "prepared_manifest.json")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("SEMIGCN_THREADS", "2")
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    stage = load_stage()
    stage.ROOT = output_root
    stage.AUTHOR = AUTHOR
    mask_records = []
    baseline_manifest = None
    same_seed_check = False
    if baseline_stage:
        baseline = output_root / baseline_stage / case
        if not (baseline / "stage_output_manifest.json").exists():
            raise FileNotFoundError(f"baseline stage missing: {baseline}")
        baseline_manifest = json.loads((baseline / "stage_output_manifest.json").read_text(encoding="utf-8"))
        # Earlier saved314 stages predate the seed field; their wrapper always
        # used the documented default314. New stages must record their seed.
        baseline_seed = int(baseline_manifest.get("seed", 314))
        same_seed_check = baseline_seed == seed
    if same_seed_check:
        # Verify the regenerated tensors before fitting. This keeps the author
        # source and its random-number consumption unchanged.
        import torch
        sys.path.insert(0, str(AUTHOR / "source"))
        from util import datamaker
        original_create = datamaker.create_dataset
        def verify_masks(*args, **kwargs):
            result = original_create(*args, **kwargs)
            mesh_dic = result[0]
            for tensor_key, filename in (("vmask_dummy", "vmask_dummy.pt"),
                                         ("fmask_dummy", "fmask_dummy.pt")):
                reference = torch.load(baseline / "dummy_mask" / filename, weights_only=True)
                actual = mesh_dic[tensor_key]
                equal = bool(torch.equal(reference.cpu(), actual.cpu()))
                ref_digest = hashlib.sha256(reference.cpu().numpy().tobytes()).hexdigest()
                digest = hashlib.sha256(actual.cpu().numpy().tobytes()).hexdigest()
                record = {"name": filename, "baseline_tensor_sha256": ref_digest,
                          "tensor_sha256": digest, "exactly_equal": equal}
                mask_records.append(record)
                print(json.dumps({"budget_mask_check": record}), flush=True)
                if not equal:
                    raise RuntimeError(f"budget rerun mask differs from baseline: {case}/{filename}")
            return result
        datamaker.create_dataset = verify_masks
    sys.argv = ["semigcn_stage.py", "--iterations", str(iterations), "--stage", stage_name, "--seed", str(seed)]
    author_started = time.perf_counter()
    stage.main()
    author_pipeline_seconds = time.perf_counter() - author_started
    manifest = output_root / stage_name / case / "stage_output_manifest.json"
    if not manifest.exists():
        raise RuntimeError(f"stage did not produce output manifest: {manifest}")
    summary = json.loads(manifest.read_text(encoding="utf-8"))
    summary["runner"] = str(Path(__file__).resolve())
    summary["device_environment"] = {"python": sys.executable, "torch": stage.torch_environment()}
    if remeasure_preparation:
        summary["runtime_measurement"] = {
            "fresh_input_preparation": True,
            "preparation_startup_inclusive_seconds": preparation_wall_seconds,
            "author_pipeline_seconds": author_pipeline_seconds,
            "input_to_restored_output_wall_seconds": time.perf_counter() - execution_started,
            "common_return_validation_seconds": None,
            "scope": "fresh damaged input through common-CGAL initialization, normalization, smoothing, author fit/refine and global restoration; selected-cap return and common safety validation recorded separately",
            "benchmark_root": str(BENCH_ROOT), "prepared_input_equivalence": equivalence,
            "prepared_input_equivalence_seconds": equivalence_seconds,
            "exclusive_device_claim": False,
        }
    if int(summary["seed"]) != seed:
        raise RuntimeError("stage recorded a different seed than the requested seed")
    if baseline_stage:
        summary["budget_comparison_baseline_stage"] = baseline_stage
        summary["budget_comparison_baseline_seed"] = baseline_seed
        summary["same_seed_mask_check_applied"] = same_seed_check
        summary["budget_same_masks"] = mask_records
        if not same_seed_check:
            summary["mask_check_skip_reason"] = "Different seeds vary the entire stochastic algorithm, including dummy masks; mask equality is not required"
        baseline_steps = int(baseline_manifest["iterations"])
        original_checkpoint = Path(baseline_manifest["refine_obj"]).parent / f"{baseline_steps}_step_0.obj"
        rerun_checkpoint = Path(summary["refine_obj"]).parent / f"{baseline_steps}_step_0.obj"
        if original_checkpoint.exists() and rerun_checkpoint.exists():
            original_digest = hashlib.sha256(original_checkpoint.read_bytes()).hexdigest()
            rerun_digest = hashlib.sha256(rerun_checkpoint.read_bytes()).hexdigest()
            summary["baseline_epoch_checkpoint"] = {
                "epoch": baseline_steps, "baseline_sha256": original_digest,
                "rerun_sha256": rerun_digest, "exactly_equal": original_digest == rerun_digest}
    manifest.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if remeasure_preparation:
        verification_started = time.perf_counter()
        command = [str(SCORING_PYTHON), str(REPO / "scripts/evaluate_external_benchmark.py"),
                   "--benchmark-root", str(BENCH_ROOT), "--semigcn-root", str(HERE),
                   "--semigcn-stage", stage_name, "--verify-learning-case", str(int(case.removeprefix("case")))]
        process = subprocess.run(command, cwd=REPO, check=False)
        summary["runtime_measurement"].update({
            "common_return_validation_startup_inclusive_seconds": time.perf_counter() - verification_started,
            "raw_input_to_verified_output_wall_seconds": time.perf_counter() - execution_started,
            "runtime_verification_returncode": process.returncode,
            "scope": "continuous fresh damaged input through common-CGAL initialization, normalization, smoothing, author fit/refine, selected-cap restoration and uncached common safety validation; includes subprocess startups and input equivalence check"})
        manifest.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if process.returncode:
            raise RuntimeError(f"learning return verification failed with code {process.returncode}")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main() -> None:
    global HERE, CPU_ROOT, BENCH_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="case021")
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--stage", default="gpu_smoke")
    parser.add_argument("--seed", type=int, default=314)
    parser.add_argument("--all", action="store_true", help="run every prepared external case")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--summary", action="store_true", help="write concise summary from gpu manifests")
    parser.add_argument("--case-protocol", type=Path, help="fixed subset protocol containing cases[].case")
    parser.add_argument("--status-file", type=Path, help="separate batch status for a budget experiment")
    parser.add_argument("--baseline-stage", help="verify exact dummy masks against this saved stage before fitting")
    parser.add_argument("--watch-budget", action="store_true", help="wait for the fixed budget batch and automatically score paired outcomes and fitting diagnostics")
    parser.add_argument("--finalize-budget", action="store_true", help="score and clean a completed fixed budget batch")
    parser.add_argument("--run-stability", action="store_true", help="run frozen seed batches sequentially and score the completed grid")
    parser.add_argument("--finalize-stability", action="store_true", help="score and append refined fitting diagnostics for a terminal stability grid")
    parser.add_argument("--after-status", type=Path, help="wait for this batch to finish before using the GPU")
    parser.add_argument("--benchmark-root", type=Path, default=BENCH_ROOT)
    parser.add_argument("--cpu-root", type=Path, help="prepared common-initializer input root")
    parser.add_argument("--output-root", type=Path, help="GPU output root; default is benchmark-root/semigcn_gpu")
    parser.add_argument("--remeasure-preparation", action="store_true", help="measure fresh raw-input CGAL preparation per case; requires one worker")
    parser.add_argument("--cleanup-scored-stage", action="store_true", help="remove regenerable stage artifacts after complete unified scoring")
    parser.add_argument("--score-report", type=Path, help="completed unified evaluation JSON required for cleanup")
    args = parser.parse_args()
    BENCH_ROOT = args.benchmark_root.resolve()
    CPU_ROOT = args.cpu_root.resolve() if args.cpu_root else BENCH_ROOT / "semigcn"
    HERE = args.output_root.resolve() if args.output_root else BENCH_ROOT / "semigcn_gpu"
    HERE.mkdir(parents=True, exist_ok=True)
    if args.remeasure_preparation and args.workers != 1:
        raise ValueError("fresh end-to-end runtime measurement requires --workers 1")
    if args.iterations < 1:
        raise ValueError("iterations must be positive")
    if not 0 <= args.seed <= 2**32 - 1:
        raise ValueError("seed must be an unsigned32-bit integer")
    if os.environ.get("PYTHONHASHSEED") != str(args.seed):
        # Hash randomization is chosen at interpreter startup, unlike the
        # explicit Python/NumPy/Torch RNGs. A fresh child makes CLI seeds exact.
        process = subprocess.run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
                                 cwd=REPO, env=dict(os.environ, PYTHONHASHSEED=str(args.seed)))
        raise SystemExit(process.returncode)
    if args.cleanup_scored_stage:
        if not args.score_report:
            raise ValueError("scored-stage cleanup requires --score-report")
        cleanup_scored_stage(args.stage, args.score_report.resolve())
        return
    if args.run_stability:
        if not args.case_protocol:
            raise ValueError("stability execution requires a frozen case protocol")
        run_stability(args.case_protocol.resolve(), args.after_status.resolve() if args.after_status else None, args.workers)
        return
    if args.finalize_stability:
        if not args.case_protocol:
            raise ValueError("stability finalization requires a frozen protocol")
        finalize_stability(args.case_protocol.resolve())
        return
    if args.watch_budget or args.finalize_budget:
        if not args.case_protocol or not args.status_file:
            raise ValueError("budget watcher requires a frozen case protocol and status file")
        if args.watch_budget:
            watch_budget(args.case_protocol.resolve(), args.status_file.resolve())
        else:
            budget_evidence(args.case_protocol.resolve(), args.status_file.resolve())
        return
    if args.all:
        if args.after_status:
            wait_for_status(args.after_status.resolve())
        batch(args.iterations, args.stage, args.workers, args.case_protocol, args.status_file, args.baseline_stage, args.seed,
              remeasure_preparation=args.remeasure_preparation)
        return
    if args.summary:
        summarize(args.stage)
        return
    if args.after_status:
        wait_for_status(args.after_status.resolve())
    run(args.case, args.iterations, args.stage, args.baseline_stage, args.seed, args.remeasure_preparation)


def batch(iterations: int, stage_name: str, workers: int, case_protocol: Path | None = None,
          status_file: Path | None = None, baseline_stage: str | None = None,
          seed: int = 314, progress_callback=None, remeasure_preparation: bool = False) -> None:
    """Launch isolated child processes, preserving one clean output per case."""
    if not 1 <= workers <= 4:
        raise ValueError("workers must be between 1 and 4")
    prepared_cases = sorted(p.parent.name for p in (CPU_ROOT / "cases").glob("case[0-9][0-9][0-9]/prepared_manifest.json"))
    source_cases = {}
    if case_protocol:
        protocol = json.loads(case_protocol.read_text(encoding="utf-8"))
        selected = [r["case"] for r in protocol["cases"]]
        if len(set(selected)) != len(selected) or not set(selected).issubset(prepared_cases):
            raise ValueError("subset protocol must contain unique prepared cases")
        cases = selected
    else:
        source_protocol = json.loads((BENCH_ROOT / "protocol.json").read_text(encoding="utf-8"))
        source_cases = {f"case{int(row['index']):03d}": row for row in source_protocol["cases"]}
        if len(source_cases) != len(source_protocol["cases"]):
            raise ValueError("benchmark protocol contains duplicate case indices")
        cases = list(source_cases)
    logs = HERE / "logs"
    logs.mkdir(exist_ok=True)
    status_path = status_file.resolve() if status_file else HERE / "batch_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if int(status.get("seed", seed)) != seed or status.get("stage", stage_name) != stage_name:
            raise RuntimeError("refusing to reuse another seed or stage's batch status")
        existing = {r["case"]: r for r in status.get("rows", [])}
    pending = []
    rows = []
    for case in cases:
        if not (CPU_ROOT / "cases" / case / "prepared_manifest.json").exists():
            failure_path = CPU_ROOT / "cases" / case / "preparation_failed.json"
            failure = json.loads(failure_path.read_text(encoding="utf-8")) if failure_path.exists() else {}
            source = source_cases.get(case, {})
            rows.append({"case": case, "seed": seed, "status": "failed", "reason": "preparation_failure",
                         "index": source.get("index"), "sample_id": source.get("sample_id"),
                         "error": failure.get("error", "Missing prepared input; generation cannot run"),
                         "preparation_failure_path": str(failure_path) if failure_path.exists() else None})
            continue
        manifest = HERE / "cases" / case / stage_name / case / "stage_output_manifest.json"
        if manifest.exists():
            saved = json.loads(manifest.read_text(encoding="utf-8"))
            if int(saved.get("seed", 314)) != seed or int(saved["iterations"]) != iterations:
                raise RuntimeError(f"saved stage belongs to a different seed or budget: {manifest}")
            if remeasure_preparation and saved.get("runtime_measurement", {}).get("runtime_verification_returncode") != 0:
                raise RuntimeError(f"saved stage lacks successful continuous fresh-input runtime measurement: {manifest}")
            row = {**existing.get(case, {}), "case": case, "seed": seed, "status": "completed",
                   "manifest": str(manifest), "reused": True}
            rows.append(row)
        else:
            pending.append(case)
            if case in existing:
                rows.append(existing[case])

    env = dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "0"), PYTHONHASHSEED=str(seed),
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
               NUMEXPR_NUM_THREADS="2", SEMIGCN_THREADS="2", PYTHONUNBUFFERED="1")

    def one(case: str) -> dict:
        started = time.perf_counter()
        log = logs / f"{case}_{stage_name}.log"
        command = [sys.executable, str(Path(__file__).resolve()), "--case", case,
                   "--iterations", str(iterations), "--stage", stage_name, "--seed", str(seed),
                   "--benchmark-root", str(BENCH_ROOT), "--cpu-root", str(CPU_ROOT), "--output-root", str(HERE),
                   "--workers", str(workers)]
        if remeasure_preparation:
            command.append("--remeasure-preparation")
        if baseline_stage:
            command.extend(["--baseline-stage", baseline_stage])
        try:
            with log.open("w", encoding="utf-8") as fh:
                proc = subprocess.run(command, cwd=REPO, env=env, stdout=fh,
                                      stderr=subprocess.STDOUT, check=False)
            manifest = HERE / "cases" / case / stage_name / case / "stage_output_manifest.json"
            state = "completed" if proc.returncode == 0 and manifest.exists() else "failed"
            return {"case": case, "seed": seed, "status": state, "returncode": proc.returncode,
                    "elapsed_seconds": time.perf_counter() - started,
                    "manifest": str(manifest), "log": str(log)}
        except Exception as exc:
            return {"case": case, "seed": seed, "status": "failed", "error": repr(exc),
                    "elapsed_seconds": time.perf_counter() - started, "log": str(log)}

    def save_status(complete):
        state = {"complete": complete, "iterations": iterations, "seed": seed,
                 "workers": workers, "stage": stage_name,
                 "benchmark_root": str(BENCH_ROOT), "cpu_root": str(CPU_ROOT), "output_root": str(HERE),
                 "fresh_preparation_measured": remeasure_preparation,
                 "case_protocol": str(case_protocol) if case_protocol else None,
                 "baseline_stage": baseline_stage, "rows": rows}
        status_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        if progress_callback:
            progress_callback(state)
    known = {r["case"] for r in rows}
    rows.extend({"case": case, "seed": seed, "status": "pending"} for case in pending if case not in known)
    save_status(False)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, case): case for case in pending}
        for future in as_completed(futures):
            row = future.result()
            rows = [r for r in rows if r.get("case") != row["case"]]
            rows.append(row)
            save_status(False)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    rows.sort(key=lambda r: r["case"])
    save_status(True)


def cleanup_scored_stage(stage_name: str, report_path: Path) -> None:
    """Delete only expendable files from this scored stage; retain raw evidence."""
    if stage_name in ("gpu100", "seed_check", "preparation_check"):
        raise ValueError("This cleanup entry point excludes historical and temporary check stages")
    if Path(stage_name).name != stage_name:
        raise ValueError("Stage must be a single directory name")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    protocol = json.loads((BENCH_ROOT / "protocol.json").read_text(encoding="utf-8"))
    if not report.get("complete") or report.get("pending_n") != 0 or report.get("semigcn_stage") != stage_name:
        raise RuntimeError("Cleanup requires completed scoring for this exact learning stage")
    if report.get("protocol_sha256") != hashlib.sha256((BENCH_ROOT / "protocol.json").read_bytes()).hexdigest():
        raise RuntimeError("Cleanup score report belongs to a different protocol")
    if Path(report.get("semigcn_root") or "").resolve() != HERE.resolve():
        raise RuntimeError("Cleanup score report belongs to a different output root")
    expected = {int(c["index"]) for c in protocol["cases"]}
    if set(report.get("original_case_indices", [])) != expected:
        raise RuntimeError("Cleanup requires all frozen protocol cases to have terminal scores")
    keep = {f"{epoch}_step_0.obj" for epoch in (100, 200, 280, 290, 300)}
    scored = {r["sample_id"]: r for r in report["rows"]}
    deleted, skipped = [], []
    for case in protocol["cases"]:
        name = f"case{int(case['index']):03d}"
        destination = (HERE / "cases" / name / stage_name / name).resolve()
        row = scored.get(case["sample_id"], {}).get("semigcn", {})
        if row.get("status") != "complete" or not (destination / "stage_output_manifest.json").exists():
            skipped.append({"case": name, "reason": "failed or unavailable fit; preserve diagnostic files"})
            continue
        if not destination.is_relative_to(HERE.resolve()) or any(x in destination.parts for x in ("seed_check", "preparation_check")):
            raise RuntimeError("Cleanup target escaped the selected formal stage")
        candidates = [p for p in (destination / "output").glob("*/*_step_*.obj") if p.name not in keep]
        candidates.extend((destination / "pooled").glob("*"))
        candidates.extend((destination / "dummy_mask").glob("*.ply"))
        candidates.append(destination / f"{name}_initial.pt")
        for path in candidates:
            if not path.is_file():
                continue
            resolved = path.resolve()
            if path.is_symlink() or not resolved.is_relative_to(destination):
                raise RuntimeError("Cleanup candidate escaped the selected case stage")
            size = resolved.stat().st_size
            resolved.unlink()
            deleted.append({"relative_path": resolved.relative_to(HERE).as_posix(), "bytes": size})
        pooled = destination / "pooled"
        if pooled.exists() and not any(pooled.iterdir()):
            pooled.rmdir()
    prior = report.get("learning_stage_cleanup", {}).get("deleted", [])
    report["learning_stage_cleanup"] = {
        "stage": stage_name, "completed": not skipped, "relative_paths_root": str(HERE),
        "deleted_files_n": len(prior) + len(deleted),
        "deleted_bytes": sum(x["bytes"] for x in prior + deleted), "deleted": prior + deleted,
        "skipped": skipped,
        "retain": ["final refine/restored_patch/returned_mesh", "100/200/280/290/300 finest checkpoints",
                   "dummy mask tensors", "full logs", "prepared input/mapping/normalization/source/seed provenance",
                   "fresh initialization and continuous runtime/native verification"]}
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"cleanup_stage": stage_name, "newly_deleted_n": len(deleted), "skipped_n": len(skipped)}), flush=True)


def wait_for_status(path: Path, progress_callback=None) -> None:
    """Queue GPU work until the preceding batch has released its workers."""
    previous = None
    while True:
        try:
            status = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(30)
            continue
        terminal = sum(r.get("status") in ("completed", "failed") for r in status.get("rows", []))
        state = (terminal, bool(status.get("complete")))
        if state != previous:
            print(json.dumps({"waiting_for_status": str(path), "terminal_n": terminal,
                              "n": len(status.get("rows", [])), "complete": state[1]}), flush=True)
            if progress_callback:
                progress_callback(status)
            previous = state
        if status.get("complete"):
            return
        time.sleep(30)


def run_stability(protocol_path: Path, after_status: Path | None, workers: int) -> None:
    """Run seed batches sequentially; preserve the single-seed budget evidence."""
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    seed_specs = protocol["seeds"]
    seeds = [int(s["seed"]) for s in seed_specs]
    expected = {r["case"] for r in protocol["cases"]}
    if len(set(seeds)) != len(seeds) or len(expected) != len(protocol["cases"]):
        raise ValueError("stability protocol needs unique seeds and unique original cases")
    if any(not 0 <= s <= 2**32 - 1 for s in seeds) or not 1 <= workers <= 4:
        raise ValueError("invalid stability seed or worker count")
    source = Path(protocol["source_protocol"])
    if hashlib.sha256(source.read_bytes()).hexdigest() != protocol["source_protocol_sha256"]:
        raise RuntimeError("stability source protocol changed")
    for case in protocol["cases"]:
        prepared = json.loads((CPU_ROOT / "cases" / case["case"] / "prepared_manifest.json").read_text(encoding="utf-8"))
        if prepared["input_sha256"] != case["input_sha256"]:
            raise RuntimeError("stability input differs from the frozen prepared mesh")
    master_path = Path(protocol.get("master_status_file", HERE / "stability_status.json"))
    phase = "waiting_for_budget" if after_status else "queued"
    active_seed = None
    waiting_state = None

    def seed_status_path(spec):
        path = Path(spec.get("status_file", HERE / f"stability_seed{spec['seed']}_status.json"))
        return path if path.is_absolute() else protocol_path.parent / path

    def refresh_master(_state=None):
        combined, summaries = [], []
        for spec in seed_specs:
            seed = int(spec["seed"])
            path = seed_status_path(spec)
            state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"rows": []}
            if int(state.get("seed", seed)) != seed or state.get("stage", spec["stage"]) != spec["stage"]:
                raise RuntimeError("stability status belongs to another seed/stage")
            by_case = {r["case"]: r for r in state.get("rows", [])}
            seed_rows = []
            for selected in protocol["cases"]:
                row = dict(by_case.get(selected["case"], {"case": selected["case"], "status": "pending"}))
                row.update({"seed": seed, "stage": spec["stage"], "index": selected["index"],
                            "sample_id": selected["sample_id"], "category": selected["category"]})
                seed_rows.append(row)
            complete_n = sum(r["status"] == "completed" for r in seed_rows)
            failed_n = sum(r["status"] == "failed" for r in seed_rows)
            combined.extend(seed_rows)
            summaries.append({"seed": seed, "stage": spec["stage"], "status_file": str(path),
                              "completed_n": complete_n, "failed_n": failed_n,
                              "pending_n": len(seed_rows) - complete_n - failed_n,
                              "complete": bool(state.get("complete", False))})
        value = {"schema": "semigcn_seed_stability_execution", "complete": phase == "complete",
                 "phase": phase, "active_seed": active_seed, "iterations": protocol["iterations"],
                 "workers": workers, "protocol": str(protocol_path), "protocol_sha256": digest,
                 "after_status": str(after_status) if after_status else None,
                 "waiting_batch": waiting_state, "expected_runs_n": len(expected) * len(seeds),
                 "completed_n": sum(r["status"] == "completed" for r in combined),
                 "failed_n": sum(r["status"] == "failed" for r in combined),
                 "pending_n": sum(r["status"] not in ("completed", "failed") for r in combined),
                 "seeds": summaries, "rows": combined}
        master_path.write_text(json.dumps(value, indent=2), encoding="utf-8")

    refresh_master()
    if after_status:
        def waiting_progress(state):
            nonlocal waiting_state
            waiting_state = {"complete": state.get("complete", False),
                             "completed_n": sum(r.get("status") == "completed" for r in state.get("rows", [])),
                             "failed_n": sum(r.get("status") == "failed" for r in state.get("rows", []))}
            refresh_master()
        wait_for_status(after_status, waiting_progress)
    for spec in seed_specs:
        if hashlib.sha256(protocol_path.read_bytes()).hexdigest() != digest:
            raise RuntimeError("frozen stability protocol changed")
        active_seed = int(spec["seed"])
        phase = "running"
        path = seed_status_path(spec)
        saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        terminal = {r["case"] for r in saved.get("rows", []) if r.get("status") in ("completed", "failed")}
        if not (saved.get("complete") and terminal == expected):
            # Only historical/default314 budget runs receive the same-mask
            # gate. Other seeds regenerate the whole stochastic algorithm.
            baseline = "gpu100" if active_seed == 314 else None
            batch(int(protocol["iterations"]), spec["stage"], workers, protocol_path,
                  path, baseline, active_seed, refresh_master)
        refresh_master()
    active_seed = None
    phase = "scoring"
    refresh_master()
    subprocess.run([sys.executable, str(Path(__file__).resolve()), "--finalize-stability",
                    "--case-protocol", str(protocol_path)], cwd=REPO, check=True)
    phase = "complete"
    refresh_master()
    print(json.dumps({"stability_execution_complete": True, "master_status": str(master_path)}), flush=True)


def finalize_stability(protocol_path: Path) -> None:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    expected = {r["case"] for r in protocol["cases"]}
    for spec in protocol["seeds"]:
        path = Path(spec["status_file"])
        path = path if path.is_absolute() else protocol_path.parent / path
        status = json.loads(path.read_text(encoding="utf-8"))
        if (not status.get("complete") or {r["case"] for r in status["rows"]} != expected or
                any(r.get("status") not in ("completed", "failed") for r in status["rows"])):
            raise RuntimeError("stability finalization requires all frozen seed/source runs to be terminal")
    subprocess.run([str(SCORING_PYTHON), str(REPO / "scripts" / "evaluate_external_benchmark.py"),
                    "--stability-protocol", str(protocol_path), "--semigcn-root", str(HERE),
                    "--output-stem", "stability_evaluation"], cwd=REPO, check=True)
    stability_fit_evidence(protocol_path)


def stability_fit_evidence(protocol_path: Path) -> None:
    """Refine every late checkpoint before comparing geometry across epochs."""
    import numpy as np
    import torch
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(AUTHOR))
    sys.path.insert(0, str(AUTHOR / "source"))
    from scripts import evaluate_external_benchmark as evaluation
    import sparse_cpu_compat
    from run_official import load_obj
    from util.mesh import Mesh
    torch.set_num_threads(2)
    Mesh.build_v2v = sparse_cpu_compat.build_v2v
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    cases = {int(c["index"]): c for c in evaluation.load(evaluation.PROTOCOL_PATH)["cases"]}
    iterations = int(protocol["iterations"])
    rows = []

    def delta_stats(values):
        values = np.asarray(values, dtype=float)
        if not len(values):
            return {"n": 0, "mean": None, "rms": None, "max": None}
        return {"n": len(values), "mean": float(np.mean(values)),
                "rms": float(np.sqrt(np.mean(values * values))), "max": float(np.max(values))}

    for spec in protocol["seeds"]:
        seed, stage_name = int(spec["seed"]), spec["stage"]
        for selected in protocol["cases"]:
            case = selected["case"]
            row = {"case": case, "index": selected["index"], "seed": seed,
                   "stage": stage_name, "status": "unavailable"}
            log = HERE / "logs" / f"{case}_{stage_name}.log"
            content = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
            losses = {int(m.group(1)): float(m.group(2))
                      for m in re.finditer(r"Epoch (\d+):[^\r\n]*loss=([0-9.eE+\-]+)", content)}
            loss_complete = sorted(losses) == list(range(1, iterations + 1))
            row["loss"] = {"complete": loss_complete, "iterations": iterations, "log": str(log),
                           "displayed_epoch_losses": [{"epoch": k, "loss": v} for k, v in sorted(losses.items())],
                           "precision": "saved tqdm display, normally three significant digits"}
            if loss_complete:
                head = float(np.mean([losses[k] for k in range(iterations - 19, iterations - 14)]))
                tail = float(np.mean([losses[k] for k in range(iterations - 4, iterations + 1)]))
                relative = (tail - head) / head
                row["loss"].update({"initial": losses[1], "final": losses[iterations],
                                    "late20_relative_change": relative,
                                    "late20_abs_change_le_5pct": abs(relative) <= .05,
                                    "late20_decline_gt_5pct": relative < -.05})
            destination = HERE / "cases" / case / stage_name / case
            manifest_path = destination / "stage_output_manifest.json"
            if not manifest_path.exists():
                row["error"] = "training failed or no completed output manifest"
                rows.append(row)
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                recorded_seed = int(manifest.get("seed", 314))
                if recorded_seed != seed:
                    raise ValueError("manifest seed differs from the frozen grid seed")
                row["seed_metadata"] = {"recorded_seed": recorded_seed,
                    "origin": "manifest" if "seed" in manifest else "historical default314 runner, fixed by the original budget protocol",
                    "python_hash_seed": manifest.get("python_hash_seed"),
                    "torch_initial_seed": manifest.get("torch_initial_seed"),
                    "seed_applications": manifest.get("seed_applications"),
                    "dummy_masks": manifest.get("dummy_masks"), "source_commit": manifest.get("source_commit")}
                mapping = np.load(destination / "restore_mapping.npz", allow_pickle=False)
                bv, bf = mapping["vertices"], mapping["faces"]
                edges = np.concatenate([bf[:, [0, 1]], bf[:, [1, 2]], bf[:, [2, 0]]])
                h = float(np.median(np.linalg.norm(bv[edges[:, 0]] - bv[edges[:, 1]], axis=1)))
                if h <= 0:
                    raise ValueError("nonpositive median input edge")
                prepared = json.loads((HERE / "cases" / case / "prepared_manifest.json").read_text(encoding="utf-8"))
                source_case = cases[int(selected["index"])]
                pred_map = evaluation.semigcn_selected_faces(source_case, evaluation.find_prediction(source_case), destination, prepared)
                if not pred_map or pred_map.get("error"):
                    raise ValueError("selected cap mapping unavailable")
                cap_ids = np.unique(pred_map["patch_faces"])
                cap_new = cap_ids[cap_ids >= len(bv)]
                cap_seam = cap_ids[cap_ids < len(bv)]
                global_ids = mapping["global_ids"]
                mesh = Mesh(str(destination / f"{case}_initial.obj"), build_mat=False)
                preserve = torch.tensor(json.loads((destination / f"{case}_vmask.json").read_text()), dtype=torch.bool)
                transform = json.loads((destination / "normalization_transform.json").read_text())
                output = Path(manifest["refine_obj"]).parent
                restored = {}
                for epoch in (iterations - 20, iterations - 10, iterations):
                    vertices, faces = load_obj(output / f"{epoch}_step_0.obj")
                    if not np.array_equal(faces, mesh.faces):
                        raise ValueError("late fitting checkpoint changed topology")
                    refined = sparse_cpu_compat.mesh_merge(mesh.Lap, mesh,
                              torch.tensor(vertices, dtype=torch.float32), preserve, w=1, w_b=0).numpy()
                    full = mapping["initializer_vertices"].copy()
                    full[global_ids] = (refined.astype(float) - np.asarray(transform["offset"])) / transform["scale"]
                    restored[epoch] = full
                final = np.load(destination / "restored_patch.npz", allow_pickle=False)["vertices"]
                parity = np.linalg.norm(restored[iterations] - final, axis=1) / h
                row["refinement_restore_max_over_edge"] = float(np.max(parity))
                if np.max(parity) > 1e-4:
                    raise ValueError("final checkpoint did not reproduce final refinement within export tolerance")
                changes = {}
                for start in (iterations - 20, iterations - 10):
                    delta = np.linalg.norm(restored[iterations] - restored[start], axis=1) / h
                    changes[f"{start}_to_{iterations}"] = {
                        "component": delta_stats(delta[global_ids]),
                        "all_retained": delta_stats(delta[:len(bv)]),
                        "selected_cap_all": delta_stats(delta[cap_ids]),
                        "selected_cap_new_vertices": delta_stats(delta[cap_new]),
                        "selected_cap_retained_seam": delta_stats(delta[cap_seam])}
                row["median_input_edge"] = h
                row["geometry_changes"] = changes
                row["late20_selected_cap_mean_le_0_01_edge"] = changes[f"{iterations - 20}_to_{iterations}"]["selected_cap_all"]["mean"] <= .01
                row["status"] = "complete"
            except Exception as exc:
                row["status"] = "error"
                row["error"] = str(exc)
            rows.append(row)
    summary = {}
    for spec in protocol["seeds"]:
        selected = [r for r in rows if r["seed"] == int(spec["seed"])]
        available = [r for r in selected if r["status"] == "complete"]
        summary[str(spec["seed"])] = {"n": len(selected), "geometry_available_n": len(available),
            "loss_complete_n": sum(r["loss"]["complete"] for r in selected),
            "late20_loss_decline_gt_5pct_n": sum(r["loss"].get("late20_decline_gt_5pct", False) for r in selected),
            "late20_loss_abs_change_le_5pct_n": sum(r["loss"].get("late20_abs_change_le_5pct", False) for r in selected),
            "late20_selected_cap_mean_le_0_01_edge_n": sum(r["late20_selected_cap_mean_le_0_01_edge"] for r in available)}
    report_path = HERE.parent / "stability_evaluation.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["fitting_trajectory_diagnostics"] = {
        "runs_n": len(rows), "source_n": len(protocol["cases"]), "seed_n": len(protocol["seeds"]),
        "geometry_available_n": sum(r["status"] == "complete" for r in rows),
        "method": "All saved280/290/300 finest checkpoints receive identical author sparse mesh_merge(w=1,w_b=0), inverse normalization and global restoration; same global median input edge across seeds",
        "interpretation": "Descriptive late-loss and motion diagnostics only. Mean cap motion <=0.01 does not prove convergence; retained seam and newly inserted cap vertex mean/RMS/max reveal localized motion. Seed variation covers the entire stochastic algorithm, not network initialization alone.",
        "per_seed": summary, "rows": rows}
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["", "## Complete fitting trajectories and refined late geometry", "",
             f"Full displayed loss histories available for {sum(r['loss']['complete'] for r in rows)}/{len(rows)} runs; refined late-geometry diagnostics available for {sum(r['status']=='complete' for r in rows)}/{len(rows)}. Distances use the same median input retained-face edge length across seeds. Raw checkpoints are never subtracted from refined final outputs.", "",
             "|Seed|Case|Late20 loss change|New cap RMS / edge|New cap max / edge|Retained seam RMS / edge|Retained seam max / edge|All selected-cap mean / edge|",
             "|---:|---|---:|---:|---:|---:|---:|---:|"]
    def fmt(value):
        return "---" if value is None else f"{value:.6f}"
    for row in rows:
        g = row.get("geometry_changes", {}).get(f"{iterations - 20}_to_{iterations}", {})
        new, seam, all_cap = g.get("selected_cap_new_vertices", {}), g.get("selected_cap_retained_seam", {}), g.get("selected_cap_all", {})
        lines.append(f"|{row['seed']}|{row['case']}|{fmt(row['loss'].get('late20_relative_change'))}|{fmt(new.get('rms'))}|{fmt(new.get('max'))}|{fmt(seam.get('rms'))}|{fmt(seam.get('max'))}|{fmt(all_cap.get('mean'))}|")
    lines += ["", "Full epoch losses (saved display precision), both280–300 and290–300 mean/RMS/max geometry changes, mask hashes and actual seed metadata are in the JSON. Newly inserted selected-cap vertices and retained seam vertices are reported separately; empty new-vertex sets remain unavailable. Low mean motion or small displayed loss changes are descriptive and do not establish optimization convergence."]
    with report_path.with_suffix(".md").open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    # Delete only regenerable files from these exact60 study stages after
    # scores and all available trajectory diagnostics have been saved.
    keep = {f"{epoch}_step_0.obj" for epoch in (100, 200, 280, 290, 300)}
    deleted, cleanup_cases = [], []
    for spec in protocol["seeds"]:
        for selected in protocol["cases"]:
            destination = (HERE / "cases" / selected["case"] / spec["stage"] / selected["case"]).resolve()
            if not destination.exists():
                continue
            if not destination.is_relative_to(HERE.resolve()) or spec["stage"] in ("gpu100", "seed_check"):
                raise RuntimeError("stability cleanup target is outside the permitted study stages")
            candidates = [p for p in (destination / "output").glob("*/*_step_*.obj") if p.name not in keep]
            candidates.extend((destination / "pooled").glob("*"))
            candidates.extend((destination / "dummy_mask").glob("*.ply"))
            candidates.append(destination / f"{selected['case']}_initial.pt")
            count, size = 0, 0
            for path in candidates:
                if not path.is_file():
                    continue
                resolved = path.resolve()
                if not resolved.is_relative_to(destination):
                    raise RuntimeError("stability cleanup file escaped its frozen case/seed stage")
                bytes_count = resolved.stat().st_size
                resolved.unlink()
                deleted.append({"relative_path": resolved.relative_to(HERE).as_posix(), "bytes": bytes_count})
                count += 1
                size += bytes_count
            pooled = destination / "pooled"
            if pooled.exists() and not any(pooled.iterdir()):
                pooled.rmdir()
            cleanup_cases.append({"case": selected["case"], "seed": spec["seed"], "stage": spec["stage"],
                                  "deleted_files_n": count, "deleted_bytes": size})
    report["cleanup"] = {"completed": True, "relative_paths_root": str(HERE),
                         "deleted_files_n": len(deleted), "deleted_bytes": sum(p["bytes"] for p in deleted),
                         "cases": cleanup_cases, "deleted_files": deleted,
                         "retain": ["100/200/280/290/300 finest checkpoints", "final refine/restored geometry and native safety audit",
                                    "full loss logs", "all prepared input/mapping/normalization provenance", "dummy-mask tensors and actual seed metadata"],
                         "excluded": ["all original gpu100 stages and retained20/50/80/90/100 checkpoints", "denied seed_check temporary stage"]}
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    with report_path.with_suffix(".md").open("a", encoding="utf-8") as fh:
        fh.write(f"\n## Post-score cleanup\n\nRemoved {len(deleted)} regenerable intermediate files ({report['cleanup']['deleted_bytes']/1024**2:.1f} MiB) from the frozen study's300-iteration stages after saving scores and fitting diagnostics. Exact deleted paths/counts are in the JSON. Retained100/200/280/290/300 finest checkpoints, final geometry/native safety results, masks, full loss logs, and input/seed/source provenance. Original100-iteration stages and the denied temporary seed_check stage were excluded.\n")
    print(json.dumps({"stability_fitting_diagnostics_runs_n": len(rows),
                      "geometry_available_n": sum(r["status"] == "complete" for r in rows)}), flush=True)


def watch_budget(protocol_path: Path, status_path: Path) -> None:
    """Finalize the already running fixed cohort without any further prompt."""
    protocol_digest = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    expected = {r["case"] for r in protocol["cases"]}
    previous = None
    while True:
        if hashlib.sha256(protocol_path.read_bytes()).hexdigest() != protocol_digest:
            raise RuntimeError("frozen budget protocol changed")
        try:
            state = json.loads(status_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(30)
            continue
        terminal = sum(r.get("case") in expected and r.get("status") in ("completed", "failed")
                       for r in state.get("rows", []))
        if terminal != previous:
            print(json.dumps({"budget_terminal_n": terminal, "n": len(expected),
                              "batch_complete": state.get("complete", False)}), flush=True)
            previous = terminal
        if state.get("complete"):
            if {r["case"] for r in state["rows"]} != expected or terminal != len(expected):
                raise RuntimeError("completed budget status does not contain the full frozen cohort")
            # Load the finalizer fresh so later reporting/cleanup fixes apply
            # without interrupting an already running training batch.
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--finalize-budget",
                            "--case-protocol", str(protocol_path), "--status-file", str(status_path)],
                           cwd=REPO, check=True)
            return
        time.sleep(30)


def budget_evidence(protocol_path: Path, status_path: Path) -> None:
    """Same-input budget comparison plus unselected loss/geometry trajectories."""
    import numpy as np
    import torch

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    status = json.loads(status_path.read_text(encoding="utf-8"))
    expected = {r["case"] for r in protocol["cases"]}
    if (not status.get("complete") or {r["case"] for r in status["rows"]} != expected or
            any(r.get("status") not in ("completed", "failed") for r in status["rows"])):
        raise RuntimeError("finalization requires all fixed budget cases to have terminal outcomes")
    report_path = HERE.parent / "budget_evaluation.json"
    protocol_digest = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    if report_path.exists():
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if (existing.get("budget_protocol_sha256") == protocol_digest and
                existing.get("cleanup", {}).get("completed") is True and
                existing.get("budget_fit_diagnostics", {}).get("analyzed_n") == len(expected)):
            print(json.dumps({"budget_report_already_finalized": str(report_path)}), flush=True)
            return
    extended, baseline = protocol["extended_stage"], protocol["baseline_stage"]
    indices = [int(r["index"]) for r in protocol["cases"]]
    evaluator_path = REPO / "scripts" / "evaluate_external_benchmark.py"
    command = [str(SCORING_PYTHON), str(evaluator_path), "--semigcn-root", str(HERE),
               "--semigcn-stage", extended, "--baseline-stage", baseline,
               "--semigcn-status-file", str(status_path), "--output-stem", "budget_evaluation",
               "--case-indices", *map(str, indices)]
    subprocess.run(command, cwd=REPO, check=True)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["n"] != len(indices) or report["pending_n"]:
        raise RuntimeError("budget evaluation did not retain every terminal cohort member")

    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(AUTHOR))
    sys.path.insert(0, str(AUTHOR / "source"))
    from scripts import evaluate_external_benchmark as evaluation
    import sparse_cpu_compat
    from run_official import load_obj
    from util.mesh import Mesh
    torch.set_num_threads(2)
    Mesh.build_v2v = sparse_cpu_compat.build_v2v
    source_cases = {int(c["index"]): c for c in evaluation.load(evaluation.PROTOCOL_PATH)["cases"]}
    rows = []

    def loss_series(case, stage_name, iterations):
        path = HERE / "logs" / f"{case}_{stage_name}.log"
        text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
        losses = {}
        for match in re.finditer(r"Epoch (\d+):[^\r\n]*loss=([0-9.eE+\-]+)", text):
            losses[int(match.group(1))] = float(match.group(2))
        complete = sorted(losses) == list(range(1, iterations + 1))
        result = {"iterations": iterations, "complete": complete,
                  "displayed_epoch_losses": [{"epoch": k, "loss": v} for k, v in sorted(losses.items())],
                  "log": str(path), "precision": "saved tqdm display, normally three significant digits"}
        if complete:
            head = float(np.mean([losses[k] for k in range(iterations - 19, iterations - 14)]))
            tail = float(np.mean([losses[k] for k in range(iterations - 4, iterations + 1)]))
            relative = (tail - head) / head
            result.update({"initial": losses[1], "final": losses[iterations],
                           "late20_relative_change": relative,
                           "late20_abs_change_le_5pct": abs(relative) <= .05,
                           "late20_decline_gt_5pct": relative < -.05})
        return result

    for selected in protocol["cases"]:
        case = selected["case"]
        row = {"case": case, "index": selected["index"], "status": "unavailable"}
        row["baseline_loss"] = loss_series(case, baseline, protocol["baseline_iterations"])
        row["extended_loss"] = loss_series(case, extended, protocol["extended_iterations"])
        b_loss, e_loss = row["baseline_loss"], row["extended_loss"]
        if b_loss["complete"] and e_loss["complete"]:
            row["loss_relative_reduction_100_to_300"] = (b_loss["final"] - e_loss["final"]) / b_loss["final"]
            row["displayed_first100_loss_prefix_equal"] = b_loss["displayed_epoch_losses"] == e_loss["displayed_epoch_losses"][:100]
        destination = HERE / "cases" / case / extended / case
        manifest_path = destination / "stage_output_manifest.json"
        if not manifest_path.exists():
            row["error"] = "extended fitting failed or did not produce a final manifest"
            rows.append(row)
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            row["mask_hashes"] = manifest.get("budget_same_masks", [])
            row["all_masks_exactly_equal"] = len(row["mask_hashes"]) == 2 and all(r["exactly_equal"] for r in row["mask_hashes"])
            row["baseline_epoch_checkpoint"] = manifest.get("baseline_epoch_checkpoint", {})
            mapping = np.load(destination / "restore_mapping.npz", allow_pickle=False)
            before_v, before_f = mapping["vertices"], mapping["faces"]
            edges = np.concatenate([before_f[:, [0, 1]], before_f[:, [1, 2]], before_f[:, [2, 0]]])
            h = float(np.median(np.linalg.norm(before_v[edges[:, 0]] - before_v[edges[:, 1]], axis=1)))
            if h <= 0:
                raise ValueError("nonpositive median input edge")
            input_manifest = json.loads((HERE / "cases" / case / "prepared_manifest.json").read_text(encoding="utf-8"))
            source_case = source_cases[int(selected["index"])]
            pred_map = evaluation.semigcn_selected_faces(source_case, evaluation.find_prediction(source_case),
                                                       destination, input_manifest)
            if not pred_map or pred_map.get("error"):
                raise ValueError("selected cap mapping unavailable")
            cap_ids = np.unique(pred_map["patch_faces"])
            cap_new = cap_ids[cap_ids >= len(before_v)]
            cap_seam = cap_ids[cap_ids < len(before_v)]
            ids = mapping["global_ids"]
            transform = json.loads((destination / "normalization_transform.json").read_text(encoding="utf-8"))
            preserve = torch.tensor(json.loads((destination / f"{case}_vmask.json").read_text()), dtype=torch.bool)
            mesh = Mesh(str(destination / f"{case}_initial.obj"), build_mat=False)
            output = Path(manifest["refine_obj"]).parent
            restored = {}
            for epoch in (80, 100, 280, 290, 300):
                positions, faces = load_obj(output / f"{epoch}_step_0.obj")
                if not np.array_equal(faces, mesh.faces):
                    raise ValueError("budget checkpoint changed topology")
                refined = sparse_cpu_compat.mesh_merge(mesh.Lap, mesh,
                           torch.tensor(positions, dtype=torch.float32), preserve, w=1, w_b=0).numpy()
                physical = (refined.astype(float) - np.asarray(transform["offset"])) / transform["scale"]
                full = mapping["initializer_vertices"].copy()
                full[ids] = physical
                restored[epoch] = full
            baseline_destination = HERE / "cases" / case / baseline / case
            baseline_manifest = json.loads((baseline_destination / "stage_output_manifest.json").read_text(encoding="utf-8"))
            baseline_positions, baseline_faces = load_obj(Path(baseline_manifest["refine_obj"]).parent / "100_step_0.obj")
            if not np.array_equal(baseline_faces, mesh.faces):
                raise ValueError("baseline100 checkpoint topology differs")
            baseline_refined = sparse_cpu_compat.mesh_merge(mesh.Lap, mesh,
                               torch.tensor(baseline_positions, dtype=torch.float32), preserve, w=1, w_b=0).numpy()
            baseline_physical = (baseline_refined.astype(float) - np.asarray(transform["offset"])) / transform["scale"]
            baseline_restored = mapping["initializer_vertices"].copy()
            baseline_restored[ids] = baseline_physical
            replay_delta = np.linalg.norm(restored[100] - baseline_restored, axis=1) / h
            row["baseline100_replay_geometry"] = {
                "comparison": "both saved100 raw checkpoints receive identical sparse mesh_merge, inverse transform and global restore",
                "component_rms_over_edge": float(np.sqrt(np.mean(replay_delta[ids] ** 2))),
                "component_max_over_edge": float(np.max(replay_delta[ids])),
                "retained_rms_over_edge": float(np.sqrt(np.mean(replay_delta[:len(before_v)] ** 2))),
                "selected_cap_rms_over_edge": float(np.sqrt(np.mean(replay_delta[cap_ids] ** 2))),
                "selected_cap_max_over_edge": float(np.max(replay_delta[cap_ids]))}
            final = np.load(destination / "restored_patch.npz", allow_pickle=False)["vertices"]
            parity = np.linalg.norm(restored[300] - final, axis=1) / h
            row["checkpoint300_restore_max_over_median_input_edge"] = float(np.max(parity))
            if np.max(parity) > 1e-4:
                raise ValueError("300 checkpoint did not reproduce final refinement within export tolerance")
            changes = {}
            def subset_motion(delta, selected_ids):
                values = delta[selected_ids]
                return {"n": len(values), "mean": float(np.mean(values)) if len(values) else None,
                        "rms": float(np.sqrt(np.mean(values * values))) if len(values) else None,
                        "max": float(np.max(values)) if len(values) else None}
            for start, end in ((80, 100), (100, 300), (280, 300), (290, 300)):
                delta = np.linalg.norm(restored[end] - restored[start], axis=1) / h
                cap = delta[cap_ids]
                changes[f"{start}_to_{end}"] = {
                    "component_rms_over_edge": float(np.sqrt(np.mean(delta[ids] ** 2))),
                    "component_max_over_edge": float(np.max(delta[ids])),
                    "retained_rms_over_edge": float(np.sqrt(np.mean(delta[:len(before_v)] ** 2))),
                    "selected_cap_mean_over_edge": float(np.mean(cap)),
                    "selected_cap_rms_over_edge": float(np.sqrt(np.mean(cap ** 2))),
                    "selected_cap_max_over_edge": float(np.max(cap)),
                    "selected_cap_new_vertices": subset_motion(delta, cap_new),
                    "selected_cap_retained_seam": subset_motion(delta, cap_seam)}
            row["geometry_changes"] = changes
            budget_change = changes["100_to_300"]["component_rms_over_edge"]
            row["baseline100_replay_rms_relative_to_budget_change"] = (
                row["baseline100_replay_geometry"]["component_rms_over_edge"] / budget_change
                if budget_change > 0 else None)
            row["median_input_edge"] = h
            row["extended_joint_descriptive_plateau"] = bool(e_loss.get("late20_abs_change_le_5pct", False) and
                                                            changes["280_to_300"]["selected_cap_mean_over_edge"] <= .01)
            row["status"] = "complete"
        except Exception as exc:
            row["status"] = "error"
            row["error"] = str(exc)
        rows.append(row)

    complete = [r for r in rows if r["status"] == "complete"]
    report["schema"] = "semigcn_fixed_cohort_fitting_budget_sensitivity"
    report["evidence_role"] = protocol["evidence_role"]
    report["budget_protocol_sha256"] = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    report["budget_fit_diagnostics"] = {
        "n": len(rows), "analyzed_n": len(complete), "normalization": "median of all input retained-face edge lengths",
        "plateau_criteria": protocol["plateau_diagnostics"],
        "criterion_interpretation": "descriptive diagnostics; no convergence guarantee, favorable-budget selection, or stopping rule",
        "mask_exact_n": sum(r.get("all_masks_exactly_equal", False) for r in rows),
        "first100_checkpoint_exact_n": sum(r.get("baseline_epoch_checkpoint", {}).get("exactly_equal", False) for r in rows),
        "extended_joint_descriptive_plateau_n": sum(r.get("extended_joint_descriptive_plateau", False) for r in rows),
        "extended_late20_loss_decline_gt_5pct_n": sum(r["extended_loss"].get("late20_decline_gt_5pct", False) for r in rows),
        "rows": rows}
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path = report_path.with_suffix(".md")
    text = md_path.read_text(encoding="utf-8")
    text = text.replace(text.splitlines()[0], "# SeMIGCN fixed 12-case 100–300 iteration budget comparison", 1)
    lines = ["", "## Fitting trajectories and same-training-path checks", "",
             "The same seed314, initialization, hyperparameters, and exact dummy masks are used. Author code does not save network/optimizer/RNG state; 300 iterations are a fresh rerun. Every original case index and all12 outcomes remain in the paired comparison. The main79 author-default100 report is unchanged.", "",
             f"Geometry diagnostics available: {len(complete)}/{len(rows)}; exact mask pairs: {report['budget_fit_diagnostics']['mask_exact_n']}/{len(rows)}; exact100 checkpoint replays: {report['budget_fit_diagnostics']['first100_checkpoint_exact_n']}/{len(rows)}.", "",
             "|Case|Loss100|Loss300|100 to300 loss reduction|Replay100 component RMS / input edge|Late20 loss change at300|Late20 selected-cap mean change / input edge|Diagnostic plateau|",
             "|---|---:|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        def fmt(v):
            return "---" if v is None else f"{v:.6f}"
        late_geometry = row.get("geometry_changes", {}).get("280_to_300", {}).get("selected_cap_mean_over_edge")
        replay = row.get("baseline100_replay_geometry", {}).get("component_rms_over_edge")
        lines.append(f"|{row['case']}|{fmt(row['baseline_loss'].get('final'))}|{fmt(row['extended_loss'].get('final'))}|{fmt(row.get('loss_relative_reduction_100_to_300'))}|{fmt(replay)}|{fmt(row['extended_loss'].get('late20_relative_change'))}|{fmt(late_geometry)}|{row.get('extended_joint_descriptive_plateau', 'unavailable')}|")
    lines += ["", "Full displayed loss histories and refinement-corrected geometry changes for80–100,100–300,280–300,290–300 are stored in the JSON, including separate mean/RMS/max for newly inserted selected-cap vertices and retained seam vertices. Low mean motion can coexist with localized motion and is not convergence. The100 replay diagnostic compares both saved100 raw checkpoints after identical refinement/restoration, reporting component/retained/selected-cap RMS and max discrepancies; compare its magnitude with100–300 changes before attributing differences solely to fitting budget. Exact file hashes alone are not used to infer material path differences. Loss precision is limited to the saved tqdm display. The plateau diagnostics compare mean281–285 versus296–300 loss (absolute relative change≤5%) and selected-cap mean refined geometry change280–300≤0.01 median input edge. These descriptive checks do not prove optimization convergence."]
    md_path.write_text(text + "\n".join(lines) + "\n", encoding="utf-8")
    # Scores and fitting diagnostics are saved before removing regenerable
    # intermediates. The fixed finest checkpoints and provenance remain.
    keep = {f"{epoch}_step_0.obj" for epoch in (100, 200, 280, 290, 300)}
    cleaned = []
    diagnostics = {r["case"]: r for r in rows}
    skipped = []
    for selected in protocol["cases"]:
        case = selected["case"]
        if diagnostics[case]["status"] != "complete":
            skipped.append({"case": case, "reason": "fitting geometry diagnostics unavailable; retain intermediate evidence for repair"})
            continue
        destination = (HERE / "cases" / case / extended / case).resolve()
        manifest_path = destination / "stage_output_manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        output = Path(manifest["refine_obj"]).parent.resolve()
        if not output.is_relative_to(destination) or not destination.is_relative_to(HERE.resolve()):
            raise RuntimeError("budget cleanup path escaped the experiment root")
        candidates = [p for p in output.glob("*_step_*.obj") if p.name not in keep]
        candidates.extend((destination / "pooled").glob("*"))
        candidates.extend((destination / "dummy_mask").glob("*.ply"))
        candidates.append(destination / f"{case}_initial.pt")
        deleted, size = 0, 0
        for path in candidates:
            if not path.is_file():
                continue
            resolved = path.resolve()
            if not resolved.is_relative_to(destination):
                raise RuntimeError("budget cleanup candidate escaped its case stage")
            size += resolved.stat().st_size
            resolved.unlink()
            deleted += 1
        pooled = destination / "pooled"
        if pooled.exists() and not any(pooled.iterdir()):
            pooled.rmdir()
        cleaned.append({"case": case, "deleted_files_n": deleted, "deleted_bytes": size})
    report["cleanup"] = {"completed": not skipped, "skipped": skipped,
                         "deleted_files_n": sum(x["deleted_files_n"] for x in cleaned),
                         "deleted_bytes": sum(x["deleted_bytes"] for x in cleaned), "cases": cleaned,
                         "retain": ["100/200/280/290/300_step_0.obj", "final refine.obj/restored_patch.npz/native selected-cap safety audit",
                                    "all prepared inputs/restoration mapping/normalization manifests", "both dummy-mask tensors", "full loss logs and source/seed provenance"]}
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    with md_path.open("a", encoding="utf-8") as fh:
        fh.write(f"\n## Cleanup\n\nRemoved {report['cleanup']['deleted_files_n']} regenerable intermediate files ({report['cleanup']['deleted_bytes']/1024**2:.1f} MiB) after saving final scores and fitting diagnostics. Retained100/200/280/290/300 finest checkpoints, final geometry and native safety results, input/mapping/normalization provenance, dummy-mask tensors, full loss logs, and seed/source metadata.\n")
    print(json.dumps({"budget_report": str(report_path), "n": len(rows),
                      "diagnostics_analyzed_n": len(complete),
                      "extended_correct_safe_n": report["summary"]["semigcn"]["correct_safe_n"],
                      "baseline_correct_safe_n": report["summary"]["semigcn_baseline"]["correct_safe_n"]}), flush=True)


def summarize(stage_name: str) -> None:
    """Persist only the key per-case provenance and integrity fields."""
    status_path = HERE / "batch_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {"rows": []}
    rows = []
    for case in sorted(p.name for p in (CPU_ROOT / "cases").glob("case[0-9][0-9][0-9]")):
        manifest = HERE / "cases" / case / stage_name / case / "stage_output_manifest.json"
        row = {"case": case, "manifest": str(manifest)}
        if manifest.exists():
            item = json.loads(manifest.read_text(encoding="utf-8"))
            row.update({key: item.get(key) for key in (
                "status", "iterations", "elapsed_seconds", "vertex_count", "face_count",
                "finite_output", "topology_preserved", "outside_component_bitwise_frozen",
                "retained_displacement_mean", "retained_displacement_max", "source_commit")})
            row["status"] = row.get("status") or "completed"
        else:
            row["status"] = "missing"
        rows.append(row)
    completed = [r for r in rows if r["status"] == "completed"]
    summary = {
        "schema": "semigcn_gpu100_summary",
        "protocol_n": len(rows),
        "completed_n": len(completed),
        "missing_or_failed_n": len(rows) - len(completed),
        "iterations": sorted({r.get("iterations") for r in completed if r.get("iterations") is not None}),
        "topology_preserved_n": sum(bool(r.get("topology_preserved")) for r in completed),
        "finite_output_n": sum(bool(r.get("finite_output")) for r in completed),
        "outside_component_bitwise_frozen_n": sum(bool(r.get("outside_component_bitwise_frozen")) for r in completed),
        "sum_case_elapsed_seconds": sum(float(r.get("elapsed_seconds") or 0.0) for r in completed),
        "max_case_elapsed_seconds": max((float(r.get("elapsed_seconds") or 0.0) for r in completed), default=0.0),
        "batch_complete": bool(status.get("complete", False)),
        "batch_workers": status.get("workers"),
        "device_environment": "myenv torch 2.10 CUDA 12.8",
        "rows": rows,
    }
    path = HERE / "gpu100_summary.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("protocol_n", "completed_n", "missing_or_failed_n", "iterations", "topology_preserved_n", "finite_output_n", "outside_component_bitwise_frozen_n", "sum_case_elapsed_seconds", "batch_complete")}, ensure_ascii=False), flush=True)
    if stage_name == "gpu100":
        fit_evidence(stage_name, rows, status)


def fit_evidence(stage_name: str, manifest_rows: list, batch_status: dict) -> None:
    """Describe the saved fitting trajectory without running new training.

    Every checkpoint receives the same sparse mesh_merge, inverse transform,
    and global restore used for refine.obj.  Raw step OBJ versus refined OBJ
    distances would mix two different representations and are not used.
    """
    import numpy as np
    import torch

    sys.path.insert(0, str(AUTHOR))
    sys.path.insert(0, str(AUTHOR / "source"))
    import sparse_cpu_compat
    from run_official import load_obj
    from util.mesh import Mesh

    torch.set_num_threads(2)
    # Apply the same memory adaptation as semigcn_stage, without touching source.
    Mesh.build_v2v = sparse_cpu_compat.build_v2v
    batch_rows = {row["case"]: row for row in batch_status.get("rows", [])}
    evidence_rows = []
    cleanup_files = []

    def distribution(values):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if not len(values):
            return {"n": 0}
        return {"n": len(values), "mean": float(values.mean()),
                "median": float(np.median(values)), "p25": float(np.percentile(values, 25)),
                "p75": float(np.percentile(values, 75)), "min": float(values.min()),
                "max": float(values.max())}

    for item in manifest_rows:
        case = item["case"]
        row = {"case": case, "status": "missing"}
        try:
            manifest_path = Path(item["manifest"])
            if not manifest_path.exists():
                evidence_rows.append(row)
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            row["official_stage_seconds"] = manifest["elapsed_seconds"]
            folder = manifest_path.parent
            output = Path(manifest["refine_obj"]).parent
            log = HERE / "logs" / f"{case}_{stage_name}.log"
            content = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
            # tqdm prints the previous value once before updating the loss.
            # The final occurrence for each epoch is its own displayed loss.
            losses = {}
            for match in re.finditer(r"Epoch (\d+):[^\r\n]*loss=([0-9.eE+\-]+)", content):
                losses[int(match.group(1))] = float(match.group(2))
            n_epochs = int(manifest["iterations"])
            row.update({"iterations": n_epochs, "log": str(log),
                        "loss_trajectory_complete": sorted(losses) == list(range(1, n_epochs + 1))})
            if row["loss_trajectory_complete"]:
                head = float(np.mean([losses[k] for k in range(81, 86)]))
                tail = float(np.mean([losses[k] for k in range(96, 101)]))
                late_change = (tail - head) / head
                row.update({"loss_displayed_by_epoch_1_to_100": [losses[k] for k in sorted(losses)],
                        "loss_initial": losses[1], "loss_at_20": losses[20],
                        "loss_at_50": losses[50], "loss_at_80": losses[80],
                        "loss_final": losses[100],
                        "loss_reduction_initial_to_final": (losses[1] - losses[100]) / losses[1],
                        "loss_mean_81_85": head, "loss_mean_96_100": tail,
                        "loss_last20_relative_change": late_change,
                        "loss_last20_abs_change_le_5pct": abs(late_change) <= 0.05,
                            "loss_last20_still_declines_gt_5pct": late_change < -0.05})
            else:
                row["loss_trajectory_limitation"] = f"Only {len(losses)}/{n_epochs} displayed epoch losses available; the standalone case021 run was not saved to a batch log. No losses imputed."

            mesh = Mesh(str(folder / f"{case}_initial.obj"), build_mat=False)
            preserve = torch.tensor(json.loads((folder / f"{case}_vmask.json").read_text()), dtype=torch.bool)
            transform = json.loads((folder / "normalization_transform.json").read_text())
            mapping = np.load(folder / "restore_mapping.npz")
            ids = mapping["global_ids"]
            selected_loop = mapping["selected_loop"]
            initial_vertices = mapping["initializer_vertices"]
            loop_vertices = initial_vertices[selected_loop]
            edge_scale = float(np.median(np.linalg.norm(loop_vertices - np.roll(loop_vertices, -1, axis=0), axis=1)))
            if edge_scale <= 0:
                raise ValueError("invalid selected boundary edge scale")
            restored = {}
            for epoch in (20, 50, 80, 90, 100):
                predicted, faces = load_obj(output / f"{epoch}_step_0.obj")
                if not np.array_equal(faces, mesh.faces):
                    raise ValueError("checkpoint topology differs from initializer")
                refined = sparse_cpu_compat.mesh_merge(
                    mesh.Lap, mesh, torch.tensor(predicted, dtype=torch.float32), preserve, w=1, w_b=0).numpy()
                physical = (refined.astype(float) - np.asarray(transform["offset"])) / transform["scale"]
                full = initial_vertices.copy()
                full[ids] = physical
                restored[epoch] = full
            final_norm, final_faces = load_obj(output / "refine.obj")
            final_saved = initial_vertices.copy()
            final_saved[ids] = (final_norm - np.asarray(transform["offset"])) / transform["scale"]
            final_npz = np.load(folder / "restored_patch.npz")["vertices"]
            if not np.array_equal(final_faces, mesh.faces):
                raise ValueError("refine topology differs from initializer")
            parity = np.linalg.norm(restored[100][ids] - final_saved[ids], axis=1) / edge_scale
            restore_parity = np.linalg.norm(final_saved - final_npz, axis=1) / edge_scale
            row.update({"selected_boundary_median_edge": edge_scale,
                        "checkpoint100_refine_reconstruction_max_edge": float(parity.max()),
                        "saved_refine_restored_npz_max_edge": float(restore_parity.max()),
                        "refinement_reconstruction_passed": bool(parity.max() <= 1e-4 and restore_parity.max() <= 1e-10)})
            if not row["refinement_reconstruction_passed"]:
                raise ValueError("checkpoint100 does not reproduce saved refinement within export tolerance")
            geometry = {}
            retained_n = len(mapping["vertices"])
            cap_ids = np.arange(retained_n, len(initial_vertices))
            for start, end in ((20, 50), (50, 100), (20, 100), (80, 100), (90, 100)):
                component_delta = np.linalg.norm(restored[end][ids] - restored[start][ids], axis=1) / edge_scale
                cap_delta = np.linalg.norm(restored[end][cap_ids] - restored[start][cap_ids], axis=1) / edge_scale
                geometry[f"{start}_to_{end}"] = {
                    "component_rms_edge": float(np.sqrt(np.mean(component_delta ** 2))),
                    "component_max_edge": float(component_delta.max()),
                    "new_cap_rms_edge": float(np.sqrt(np.mean(cap_delta ** 2))),
                    "new_cap_max_edge": float(cap_delta.max())}
            row["geometry_changes"] = geometry
            # This is a transparent diagnostic threshold, not a stopping rule.
            row["late20_component_rms_le_0_05_edge"] = geometry["80_to_100"]["component_rms_edge"] <= 0.05
            row["late20_loss_and_geometry_diagnostic_plateau"] = bool(
                row.get("loss_last20_abs_change_le_5pct", False) and row["late20_component_rms_le_0_05_edge"])
            # The tqdm loop excludes data creation/model construction and the
            # final mesh_merge. It includes periodic test forward/OBJ exports.
            loop_times = re.findall(r"Epoch 100:[^\r\n]*100/100 \[([0-9:]+)<", content)
            if loop_times:
                parts = [float(x) for x in loop_times[-1].split(":")]
                loop_seconds = sum(value * 60 ** i for i, value in enumerate(reversed(parts)))
                row["training_loop_seconds_from_tqdm"] = loop_seconds
                row["official_stage_seconds"] = manifest["elapsed_seconds"]
                row["data_model_refinement_restore_seconds_combined"] = max(0.0, float(manifest["elapsed_seconds"]) - loop_seconds)
            batch_case = batch_rows.get(case, {})
            if "elapsed_seconds" in batch_case:
                row["child_pipeline_seconds"] = batch_case["elapsed_seconds"]
                row["import_smoothing_launch_seconds_combined"] = max(0.0, float(batch_case["elapsed_seconds"]) - float(manifest["elapsed_seconds"]))
            row["status"] = "complete"
            keep_steps = {f"{epoch}_step_0.obj" for epoch in (20, 50, 80, 90, 100)}
            removable = [p for p in output.glob("*_step_*.obj") if p.name not in keep_steps]
            removable.extend((folder / "pooled").glob("*"))
            removable.extend((folder / "dummy_mask").glob("*.ply"))
            # initial.pt is a cached graph. Inputs, masks, hashes, source and
            # RNG seed suffice to regenerate it; keep all masks as evidence.
            removable.append(folder / f"{case}_initial.pt")
            for path in removable:
                if path.is_file():
                    cleanup_files.append({"relative_path": path.relative_to(HERE).as_posix(), "bytes": path.stat().st_size})
        except Exception as exc:
            row["status"] = "error"
            row["error"] = str(exc)
        evidence_rows.append(row)
        print(json.dumps({"fit_evidence_case": case, "status": row["status"], "error": row.get("error")}), flush=True)

    good = [r for r in evidence_rows if r["status"] == "complete"]
    loss_good = [r for r in good if r.get("loss_trajectory_complete")]
    loss_summary = {
        "available_trajectory_n": len(loss_good), "missing_trajectory_n": len(good) - len(loss_good),
        "initial_to_final_reduction": distribution([r["loss_reduction_initial_to_final"] for r in loss_good]),
        "last20_relative_change": distribution([r["loss_last20_relative_change"] for r in loss_good]),
        "last20_abs_change_le_5pct_n": sum(r["loss_last20_abs_change_le_5pct"] for r in loss_good),
        "last20_still_declines_gt_5pct_n": sum(r["loss_last20_still_declines_gt_5pct"] for r in loss_good)}
    geometry_summary = {pair: {metric: distribution([r["geometry_changes"][pair][metric] for r in good])
                               for metric in ("component_rms_edge", "component_max_edge", "new_cap_rms_edge", "new_cap_max_edge")}
                        for pair in ("20_to_50", "50_to_100", "20_to_100", "80_to_100", "90_to_100")}
    timing_fields = ("training_loop_seconds_from_tqdm", "official_stage_seconds", "data_model_refinement_restore_seconds_combined", "child_pipeline_seconds", "import_smoothing_launch_seconds_combined")
    evidence = {
        "schema": "semigcn_saved_fit_evidence", "stage": stage_name,
        "protocol_n": len(evidence_rows), "analyzed_n": len(good), "complete": len(good) == len(evidence_rows),
        "iterations": 100, "source_commit": "fe17377db287298c77f325eeeeec5285b1c0f626",
        "method": "Same saved seed314 runs; no additional training. For each step_0: same sparse mesh_merge(w=1,w_b=0), inverse normalization, and restore_mapping as final refinement. Distances normalized by selected boundary median edge; component and all newly added cap vertices (both initializer holes) reported separately. These fitting diagnostics are not target-specific quality scores.",
        "official_default_iterations": 100,
        "official_default_evidence": "baselines/semigcn/source/mgcn_wo_gt.py get_parser: parser.add_argument('-iter', type=int, default=100)",
        "loss_precision": "tqdm displayed loss, normally three significant digits; full precision scalar histories were not saved",
        "diagnostic_thresholds": {"absolute_last20_mean_loss_change": 0.05, "late20_component_rms_boundary_edge": 0.05},
        "threshold_interpretation": "Descriptive thresholds only, not validation-tuned and not used for training termination or baseline selection. A 100-epoch author-default budget does not by itself prove convergence.",
        "loss": loss_summary, "geometry": geometry_summary,
        "joint_diagnostic_plateau_n": sum(r["late20_loss_and_geometry_diagnostic_plateau"] for r in loss_good),
        "joint_diagnostic_available_n": len(loss_good),
        "late20_component_rms_le_0_05_edge_n": sum(r["late20_component_rms_le_0_05_edge"] for r in good),
        "majority_reaches_joint_diagnostic_plateau": sum(r["late20_loss_and_geometry_diagnostic_plateau"] for r in loss_good) > len(loss_good) / 2,
        "interpretation": "All 79 completed the author-default 100-epoch budget with substantial fitting. Most saved trajectories still improve in the final window; neither optimization convergence nor geometry convergence is established. This is a default-budget comparison, not a fully converged baseline claim.",
        "refinement_reconstruction_passed_n": sum(r["refinement_reconstruction_passed"] for r in good),
        "timing": {key: distribution([r[key] for r in good if key in r]) for key in timing_fields},
        "timing_limitations": "Stage timing starts before official.main and includes graph/dummy-mask/model construction, fitting, checkpoint exports, final mesh_merge, restore and hashes. It excludes the common-CGAL input preparation and initial smoothing. Batch-child timing includes interpreter/imports and smoothing as well; subtraction cannot isolate smoothing or refinement alone. Three workers shared one GPU; these are wall times under contention, not single-job latency.",
        "cleanup": {"deleted": False, "relative_paths_root": str(HERE), "candidate_files_n": len(cleanup_files), "candidate_bytes": sum(p["bytes"] for p in cleanup_files),
                    "retain": ["all prepared inputs/manifests, normalization_transform.json, restore_mapping.npz", "refine.obj, restored_patch.npz, stage_output_manifest.json, unified scores", "20/50/80/90/100_step_0.obj, loss logs, fit_evidence.json/md", "vmask_dummy.pt and fmask_dummy.pt, source commit and seed314 runner"],
                    "candidate_files": cleanup_files,
                    "reason": "Other checkpoints/coarse checkpoints, pooled intermediate meshes, colored mask preview PLY and initial graph cache are regenerable and are unused by final scoring. Await final unified scoring before deleting. CPU diagnostic stages/smoke stages can be removed by root after checking no final score references them."},
        "rows": evidence_rows}
    (HERE / "fit_evidence.json").write_text(json.dumps(evidence, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# SeMIGCN saved fitting evidence", "", f"Geometry analyzed for {len(good)}/{len(evidence_rows)} saved 100-epoch runs; complete displayed loss histories for {len(loss_good)}/{len(evidence_rows)} (case021 standalone log was not saved). No new training and no loss values imputed.", "",
             "Checkpoint geometry is compared only after the same sparse mesh_merge, inverse normalization and global restoration used for the final output. The 100-epoch reconstruction is checked against saved refine.obj and restored_patch.npz. New-cap vertices here include both initializer holes; target-specific evaluation is separate.", "",
             f"- Refinement reconstruction passed: {evidence['refinement_reconstruction_passed_n']}/{len(good)}.",
             f"- Median initial-to-final loss reduction: {loss_summary['initial_to_final_reduction'].get('median', 0):.1%}.",
             f"- Absolute last-20 loss change at most 5%: {loss_summary['last20_abs_change_le_5pct_n']}/{len(loss_good)}.",
             f"- Loss still decreases by more than 5% in the last 20 epochs: {loss_summary['last20_still_declines_gt_5pct_n']}/{len(loss_good)}.",
             f"- Both descriptive loss and geometry plateau diagnostics: {evidence['joint_diagnostic_plateau_n']}/{len(loss_good)}.", "",
             "| Saved epoch pair | Component RMS / boundary edge, median | New-cap RMS / boundary edge, median |", "|---|---:|---:|"]
    for pair, stats in geometry_summary.items():
        lines.append(f"| {pair.replace('_to_', ' to ')} | {stats['component_rms_edge'].get('median', 0):.6f} | {stats['new_cap_rms_edge'].get('median', 0):.6f} |")
    lines += ["", "Loss values have the precision of tqdm's saved display (normally three significant digits). Last-20 change compares mean loss at epochs81–85 with epochs96–100. Plateau thresholds are descriptive, not a stopping criterion. Continued final-window loss or geometry change is evidence against claiming full convergence, even though all runs use the author's default 100-epoch budget.", "", "**Interpretation:** " + evidence["interpretation"], "", "## Timing scope", "", evidence["timing_limitations"], "", "| Timing quantity | Cases | Median seconds |", "|---|---:|---:|"]
    for key, stats in evidence["timing"].items():
        lines.append(f"| {key} | {stats['n']} | {stats.get('median', 0):.2f} |")
    lines += ["", "## Proposed cleanup", "", f"No files deleted. {len(cleanup_files)} regenerable intermediate files ({sum(p['bytes'] for p in cleanup_files) / 1024**2:.1f} MiB) can be deleted after final unified scores are saved. The exact paths are in fit_evidence.json.", "", "Retain final outputs, prepared input and restoration provenance, dummy mask tensors and seeds, full loss logs, and the 20/50/80/90/100 finest-resolution checkpoints used for this evidence. Other checkpoints, coarse-resolution checkpoints, pooled meshes, colored dummy-mask previews, and cached initial.pt are disposable."]
    (HERE / "fit_evidence.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"fit_evidence_analyzed_n": len(good), "protocol_n": len(evidence_rows), "joint_diagnostic_plateau_n": evidence["joint_diagnostic_plateau_n"], "loss_last20_still_declines_gt_5pct_n": loss_summary["last20_still_declines_gt_5pct_n"]}), flush=True)


if __name__ == "__main__":
    main()
