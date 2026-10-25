"""Portable entry point for unchanged frozen inference and source-level scoring."""
from pathlib import Path
import argparse, hashlib, json, os, sys, time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

def rim(loop):
    return {tuple(sorted((int(a), int(b)))) for a, b in zip(loop, list(loop[1:])+[loop[0]])}

def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")

def score(before, faces, target, loop, candidate):
    from scripts.evaluate_external_benchmark import geometry_metrics
    import numpy as np
    if candidate is None:
        return {"accepted": False, "correct_safe": False}
    with np.load(candidate["patch_path"], allow_pickle=False) as data:
        vertices, patch = data["vertices"], data["patch_faces"]
    geom = geometry_metrics(before, faces, vertices, patch, loop)
    target_match = rim(loop.tolist()) == rim(target)
    safe = (candidate.get("verification_status") == "success"
            and candidate.get("strict_selected_loop_closure") is True
            and candidate.get("introduced_intersection") is False
            and geom.get("geometry_no_new_topology_defect") is True)
    return {"accepted": True, "correct_safe": bool(safe and target_match),
            "selected_target": target_match, "native_safe": bool(safe),
            "chosen_method": candidate["method"], "patch_path": candidate["patch_path"],
            "geometry": geom}

def run_case(case, output, workers):
    import numpy as np
    from tcmr import integrated_repair as frozen
    from tcmr.boundary import extract_boundary_loops
    from scripts.local_repair_topology import inspect_target_cycle
    inp = ROOT / case["input_path"]
    if hashlib.sha256(inp.read_bytes()).hexdigest() != case["input_sha256"]:
        raise ValueError("Fixed input checksum differs: " + case["sample_id"])
    with np.load(inp, allow_pickle=False) as data:
        vertices, faces, cue = data["vertices"], data["faces"], data["cue"]
    work = output / (case["cohort"] + "_" + str(case["index"]).zfill(3))
    # Only these three observed quantities enter the frozen method.
    started = time.perf_counter()
    prediction = frozen.infer(vertices, faces, cue, work, workers=workers)
    inference_seconds = time.perf_counter() - started
    decision = prediction["decisions"]["prior_locked_pool_with_joint"]
    selected = next((r for r in prediction["loops"] if r["loop_id"] == decision.get("selected_loop_id")), None)
    tcmr_candidate = next((c for c in selected["candidates"] if c["method"] == decision.get("chosen_method")), None) if selected else None
    tcmr_loop = np.load(selected["input_path"])["selected_loop"] if selected else None
    # The comparison baseline independently considers all eligible input loops.
    loops = extract_boundary_loops(vertices, faces)
    choices = [(float(np.linalg.norm(vertices[l].mean(0)-cue[:3])), i, l)
               for i, l in enumerate(loops) if inspect_target_cycle(faces,l)["input_simple_closed_boundary"]]
    cgal_candidate, cgal_loop, cgal_id = None, None, None
    if choices:
        _, cgal_id, cgal_loop = min(choices, key=lambda x:(x[0],x[1]))
        candidates = []
        for row in prediction["loops"]:
            saved_loop = np.load(row["input_path"])["selected_loop"]
            if rim(saved_loop.tolist()) == rim(cgal_loop.tolist()):
                candidates = [c for c in row["candidates"] if c["method"] in ("cgal_refine_fair", "cgal_polyline")]
                break
        if not candidates:
            baseline_dir = work / "independent_cgal"
            baseline_dir.mkdir()
            baseline_input = baseline_dir / "input.npz"
            np.savez_compressed(baseline_input, vertices=vertices, faces=faces, selected_loop=cgal_loop)
            candidates = [frozen._run_candidate({"method":m, "input_path":str(baseline_input.resolve()),
                "input_geometry_key":frozen.content_key(vertices,faces,cgal_loop)}, baseline_dir/(m+".json"))
                for m in ("cgal_refine_fair", "cgal_polyline")]
        eligible = [c for c in candidates if c.get("safe_eligible") is True
                    and c.get("seam_normal_error") is not None and np.isfinite(c["seam_normal_error"])]
        cgal_candidate = min(eligible,key=lambda c:(c["seam_normal_error"],c["method"])) if eligible else None
    # Target labels below are used only after both independent decisions exist.
    result = {"cohort":case["cohort"],"index":case["index"],"sample_id":case["sample_id"],
        "source_model_id":case["source_model_id"],"input_sha256":case["input_sha256"],
        "tcmr_inference_seconds":inference_seconds,"target_labels_passed_to_inference":False,
        "tcmr":score(vertices,faces,case["target_loop_vertices"],tcmr_loop,tcmr_candidate),
        "cgal_pool":score(vertices,faces,case["target_loop_vertices"],cgal_loop,cgal_candidate)}
    result["tcmr"]["selected_loop_id"] = decision.get("selected_loop_id")
    result["cgal_pool"]["selected_loop_id"] = cgal_id
    write(work/"portable_scoring.json",result)
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort",choices=("holdout50","cross_category60"),default="holdout50")
    parser.add_argument("--index",type=int,default=0)
    parser.add_argument("--all",action="store_true",help="Run every case in the selected cohort")
    parser.add_argument("--output",type=Path,default=ROOT/"outputs")
    parser.add_argument("--workers",type=int,choices=(1,2),default=1)
    parser.add_argument("--cgal-path",type=Path,help="Existing compatible CGAL wheel directory; inherited by native child processes")
    args = parser.parse_args()
    if args.cgal_path:
        cgal = args.cgal_path.resolve()
        if not (cgal/"CGAL").is_dir():
            raise ValueError("--cgal-path must contain the CGAL package directory")
        sys.path.insert(0,str(cgal))
        os.environ["PYTHONPATH"] = os.pathsep.join([str(cgal),str(ROOT),os.environ.get("PYTHONPATH","")])
    else:
        os.environ["PYTHONPATH"] = os.pathsep.join([str(ROOT),os.environ.get("PYTHONPATH","")])
    manifest = json.loads((ROOT/"portable_inputs.json").read_text(encoding="utf-8"))
    cases = [c for c in manifest["cases"] if c["cohort"] == args.cohort
             and (args.all or c["index"] == args.index)]
    if not cases:
        raise ValueError("No fixed case matches this index/cohort")
    output = args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for case in cases:
        rows.append(run_case(case,output,args.workers))
        write(output/(args.cohort+"_summary.json"),{"complete":len(rows)==len(cases),"rows":rows,
            "counts":{m:{"n":len(rows),"correct_safe_n":sum(r[m]["correct_safe"] for r in rows)} for m in ("tcmr","cgal_pool")}})
        print(json.dumps({"case":case["sample_id"],"completed":len(rows),"planned":len(cases),
            "correct_safe":{m:rows[-1][m]["correct_safe"] for m in ("tcmr","cgal_pool")}}),flush=True)

if __name__ == "__main__":
    main()
