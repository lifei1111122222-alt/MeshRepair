# Local mesh repair with bounded interface refinement

This repository contains the frozen mesh-repair method and the essential experiment and figure scripts for the accompanying paper. The method selects a requested boundary, constructs candidate patches, jointly refines the patch and a small retained collar, and returns an eligible repair after geometric verification. The source release consolidates the implementation and removes development-only entrypoints without changing the frozen algorithm or model weights.

## Quick start

Run commands from the repository root in an environment with the packages recorded in `requirements.txt`. The recorded inference environment uses Windows, Python 3.9.1 and `D:/miniconda/python.exe`; CGAL bindings `6.0.1.post202410241521` are installed separately. Native binary compatibility must be checked when installing on another platform. Optional baseline and rendering dependencies are listed in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

The portable manifest supplies **110 fixed damaged inputs**: 50 independent holdout sources and 60 cross-category sources. Frozen weights are in `models/cbe.pt` and `models/ranker.joblib`. Start with one source and a fresh output directory:

```powershell
python run_reproduction.py --cohort holdout50 --index 0 --output outputs/holdout_check --cgal-path .external_baselines/cgal601
```

`--cgal-path` points to the directory containing the `CGAL` package. It can be omitted when compatible bindings are already discoverable by the interpreter or installed at the default local location. On the original host, `python` in these commands can be replaced with `D:/miniconda/python.exe`.

To process both fixed cohorts:

```powershell
python run_reproduction.py --cohort holdout50 --all --output outputs/holdout50 --cgal-path .external_baselines/cgal601
python run_reproduction.py --cohort cross_category60 --all --output outputs/cross_category60 --cgal-path .external_baselines/cgal601
```

A one-source run from a separate release directory reproduced the selected loop, candidate and patch connectivity, with a maximum coordinate difference of `2.13e-14`; both the method and CGAL passed the correct-safe criterion. This check covers one source, not a new full 110-source run.

The runner checks input checksums, saves candidates and verification results, and writes per-source scores and a cohort summary. It evaluates the frozen method and an independent cue-nearest CGAL pool. Target annotations are read only for scoring. Use a new output directory for each repetition; defaults use one worker, with `--workers 2` available.

## Repository layout

| Location | Contents |
|---|---|
| `run_reproduction.py`, `portable_inputs.json` | Portable inference and evaluation on the 110 fixed inputs |
| `tcmr/` | Boundary features, frozen inference, joint refinement and geometry checks |
| `models/` | Frozen boundary encoder and candidate ranker |
| `scripts/` | Essential formal comparisons, scoring, training support and paper figures |
| `baselines/semigcn/` | Preserved author source and the explicitly adapted learning-baseline runner |
| `data/`, `experiments/` | Fixed inputs, protocols and retained experimental evidence, according to the distributed package |
| `release_manifest.json` | Current source inventory, release hashes and module-renaming record |

The convenient Python API is `tcmr.final_method.infer(vertices, faces, cue, work_dir)`. Its returned `decision` is the frozen final policy. `tcmr.integrated_repair.infer` also exposes all candidate and policy records.

## Formal experiments and figures

The portable runner is the starting point for fresh inference. The formal experiment drivers and plotting scripts additionally require the original saved candidate meshes, reference assets and result caches described in [REPRODUCIBILITY.md](REPRODUCIBILITY.md). Their presence in the source release does not make every historical experiment self-contained.

Existing result paths and original source hashes remain historical provenance. Source consolidation changes file hashes, so original hash-locked protocols cannot be resumed directly with renamed release files. Use the portable runner for the fixed inputs, or prepare a separate new protocol and output directory for a broader rerun.

For paper figures, select existing results and configure your own output and licensed font locations:

```powershell
$env:TCMR_FIGURE_DIR = "D:/paper_figures"
$env:TCMR_PREVIEW_DIR = "D:/paper_figure_previews"
$env:TCMR_HELVETICA_DIR = "D:/fonts/Helvetica"
python scripts/generate_paper_quant_figures.py --only submission-additions
python scripts/generate_paper_mesh_figures.py --only collar
python scripts/generate_paper_method_figure.py
python scripts/generate_method_target_and_refinement.py
```

Figures use Helvetica and the manuscript palette `#E38067`, `#271A92`, `#724BB6`, `#6F15AC`, `#A730A2`. Fonts are supplied separately by the user.

## Attribution

The SeMIGCN author snapshot retains its MIT license and copyright notice in `baselines/semigcn/source/LICENSE` (Shota Hattori, 2024), with its source manifest and README. The local baseline uses an explicit common-CGAL initialization and selective-repair adaptation. The variational attachment comparison is an independent equation-based adaptation. Dataset, font and third-party library licenses continue to apply to their respective assets.
