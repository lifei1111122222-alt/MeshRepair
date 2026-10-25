# Reproducing the paper results

## Frozen method and release scope

The source release preserves the frozen inference algorithm, configuration and two model weights. It removes unused development entrypoints, consolidates shared helpers and replaces versioned Python module names. `release_manifest.json` describes the current release; `experiments/frozen_method.json`, when distributed with the evidence, records the original method and historical source hashes. Their hashes refer to different source layouts and must remain distinct.

The portable runner uses the 50 holdout and 60 cross-category COSEG sources indexed in `portable_inputs.json`. It receives only damaged vertices, faces and the spatial cue during inference. Saved target annotations enter evaluation after the method and independent CGAL comparator have selected their outputs. This reproduces inference and shared local metrics from the archived weights. Reference-surface distances and the broader formal studies also require their original reference meshes and caches.

The frozen method evaluates at most two ranked loops with CGAL refine/fair, CGAL polyline, center-fan and bounded joint-refinement candidates. Joint refinement uses float64 Adam, 10 updates, learning rate 0.02 and a two-ring retained collar. Each retained vertex is constrained to 0.02 times the original boundary median edge length; the exterior and connectivity are fixed. The returned policy is `prior_locked_pool_with_joint`. Optional experimental branches in shared geometry modules are not enabled by this policy.

A candidate is eligible only after successful native verification, strict selected-loop closure and explicitly no newly introduced intersection. Unknown, failed and timed-out checks are ineligible. Selection stays within the locked target and uses the lowest eligible seam score. An empty eligible set yields abstention. The portable summary retains the verification records and geometry measurements needed to inspect each decision.

## Run and interpret the output

Follow the commands in [README.md](README.md). For example, on the original host:

```powershell
D:/miniconda/python.exe run_reproduction.py --cohort holdout50 --index 0 --output outputs/check --cgal-path .external_baselines/cgal601
```

The CGAL argument names a directory containing `CGAL/`. Native construction and verification execute in separate child processes because the original Windows environment requires isolation from Torch. Environment and native-wheel versions can affect floating-point results.

The independent CGAL comparator considers all eligible input boundaries and selects the nearest boundary to the cue without learned ranking. It reuses an identical generated CGAL candidate when available and generates missing candidates. This supports a matched geometry comparison. The portable runner's TCMR inference timing and reused baseline candidates do not provide a new independent end-to-end timing table.

A one-source reproduction was completed from a separate release directory. The selected target was loop 1, the selected candidate was `bounded_joint10`, and patch connectivity matched the historical output. The maximum absolute coordinate difference was `2.1316282072803006e-14`; both the method and independent CGAL were correct-safe. This is an executed single-source check. The release does not report a new full 110-source rerun or new model training.

## Essential workflows and retained evidence

| Workflow | Entry points | Required evidence or inputs |
|---|---|---|
| Fixed-input inference | `run_reproduction.py` | `portable_inputs.json`, its 110 NPZ files and `models/` |
| External outcomes and uniform runtime | `formal_coseg_external_benchmark.py`, `evaluate_external_benchmark.py` | Original holdout/cross-category protocols, predictions and baseline outputs |
| Collar controls, seam ablation and fallback | `evaluate_collar_controls.py` | `experiments/formal_collar_mechanism/` and referenced original candidate caches |
| Equal-motion and nearest-method comparisons | `formal_movable_neighborhood.py` | `experiments/formal_movable_neighborhood/` and common initializers |
| Reference geometry | `evaluate_reference_surface.py` | Original COSEG meshes, deletion metadata and saved output surfaces |
| Scans | `scan_edit_benchmark.py`, `real_scan_external_benchmark.py` | Recorded scan protocols, raw/reference archives and saved outputs |
| Cached displacement sensitivity | `validation_final_sensitivity.py` | `experiments/validation_final_sensitivity/` and its historical meshes |
| Learned baseline | `formal_semigcn_external_benchmark.py`, `run_semigcn.py` | Common-initializer cases, budget/seed protocols and isolated baseline environment |
| Paper figures | The four figure-generation scripts | Recorded result JSONs, selected mesh outputs and Helvetica |

Entry points in this table are under `scripts/` unless shown at the repository root. Use their `--help` options to select the intended action. Several drivers generate new outputs; calling a driver is not necessarily a read-only summary operation.

Original `data/` and `experiments/` paths retain historical version tokens because saved protocols and per-source records refer to them. Those identifiers describe provenance. The released Python source names are version-free. Hash-locked historical protocols will reject renamed code: start a fresh protocol/output location for new formal runs, and keep existing measurements unchanged.

The primary result files are `unified_evaluation.json` in the holdout and cross-category experiment directories; `results.json`, `fairing_fallback_results.json`, `predicted_fallback_results.json` and `seam_ablation_results.json` in the collar-mechanism directory; and `bounded_fair_results.json`, `attachment_results.json` and `attachment_adequacy.json` in the movable-neighborhood directory. Scan evidence is recorded in `scan_robustness/report.json` and `real_scan_external/{unified_results,independent_cgal_results}.json`. A compact upload may retain these reports without the large geometry caches needed to regenerate every panel or baseline.

## Dependencies and baseline adaptation

`requirements.txt` records the verified inference and shared-scoring packages for the original Python 3.9.1 environment. Install CGAL `6.0.1.post202410241521` separately. Optional measured dependencies are `pymeshfix 0.17.2`, `fast-simplification 0.1.12`, `pyvista 0.46.5`, `VTK 9.5.2`, `matplotlib 3.9.4` and `Pillow 11.3.0`. Natural-scan reference extraction also requires `7z` and the original archives.

Bounded biharmonic fairing used a separate Python 3.12.14 environment with `clarabel 0.11.1` and `scipy 1.16.3`; select its interpreter with `TCMR_CONVEX_PYTHON`. The official MeshLab normal-filter comparison used `pymeshlab 2021.10`; select its compatible interpreter with `TCMR_MESHLAB_PYTHON`. Both variables default to the current interpreter, which must contain the respective dependencies. The variational attachment comparator retains its independent-adaptation label and recorded 200-iteration budget.

The recorded SeMIGCN GPU environment used Python 3.10.19, `torch 2.10.0+cu128`, `torch-geometric 2.8.0.post1`, `torch_scatter 2.1.2+pt210cu128`, `numpy 2.2.6`, `scipy 1.15.3`, `scikit-learn 1.7.2`, `pymeshlab 2021.10`, `opencv-python 4.13.0.92` and `tqdm 4.67.3`. The upstream dependency files in `baselines/semigcn/source/` describe the author setup rather than this measured environment. Run `scripts/run_semigcn.py` using its compatible isolated interpreter and set `SEMIGCN_SCORING_PYTHON` when scoring should use the inference environment. Its `--benchmark-root`, `--cpu-root` and `--output-root` arguments select prepared cases and destinations. `--iterations 300 --seed 314` describes the formal fitting setting; budget and stability experiments use their saved protocols. The author snapshot is pinned to commit `fe17377db287298c77f325eeeeec5285b1c0f626`, and its MIT license, copyright and README are preserved. Common-CGAL initialization, target selection and sparse-memory compatibility are explicit local adaptations.

## Figures and interpretation

`TCMR_FIGURE_DIR` selects PDF output, `TCMR_PREVIEW_DIR` selects previews, and `TCMR_HELVETICA_DIR` supplies licensed Helvetica font files. The figure scripts read saved measurements and actual meshes. `experiments/paper_figure_source_data.json` records selected sources, measurements and rendering details. Full rendering needs the referenced geometry caches; the fixed 110 damaged inputs alone do not include all compared output meshes or reference surfaces.

Source objects are the independent statistical units. The collar and nearest-method studies reuse the same 110 COSEG sources; repeated scan conditions are grouped by their source object. Paired geometry comparisons use their reported common-valid denominators. The independent natural-hole comparison contains 14 sources, with 13 common correct-safe same-target geometry pairs. Matched end-to-end runtime and optimizer-only timing have different scopes.

The frozen CBE weights support inference, but its exact historical training recipe and curve are unavailable. Current encoder training code cannot recover that provenance. The ranker recipe is recorded, while historical PartNet train/validation/test annotations have overlapping raw source identities; the independent COSEG cohorts provide the separate-source evaluation. Cached displacement sensitivity retains its documented incomplete generation provenance and should be interpreted accordingly.
