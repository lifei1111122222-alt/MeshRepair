"""Render reproducible qualitative mesh figures from frozen experiment outputs.

The sources are the saved external COSEG, cross-category, Stanford scan-edit,
natural-scan, independent holdout, and scan-perturbation experiments.  No mesh
is inferred or repaired by this script.  Every highlighted triangle is read
from a saved result (or identified as a new face in a saved complete mesh).
Run: python scripts/generate_paper_mesh_figures.py
For the matched collar mechanism plate only, use --only collar.
For temporary visual QA, set PAPER_MESH_PREVIEW=1 to also write PNGs under
experiments/paper_figure_previews; remove that directory after inspection.
"""

from __future__ import annotations

import json
import os
import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import pyvista as pv


ROOT = Path(__file__).resolve().parents[1]
PAPER = Path(os.environ.get("TCMR_FIGURE_DIR", str(ROOT / "figures")))
PREVIEW = Path(os.environ.get("TCMR_PREVIEW_DIR", str(ROOT / "figures")))
WRITE_PREVIEWS = os.environ.get("PAPER_MESH_PREVIEW", "0") == "1"

# The user supplied licensed Helvetica fonts, installed in this user-local folder.
FONT_DIR = Path(os.environ.get("TCMR_HELVETICA_DIR", str(Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "HelveticaFonts")))
for font_path in sorted(FONT_DIR.glob("*.ttf")):
    font_manager.fontManager.addfont(str(font_path))
font_manager.findfont(font_manager.FontProperties(family="Helvetica"), fallback_to_default=False)
plt.rcParams.update({
    "font.family": "Helvetica",
    "font.size": 12,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Helvetica",
    "mathtext.it": "Helvetica:italic",
    "mathtext.bf": "Helvetica:bold",
    "pdf.fonttype": 42,
    "text.color": "black",
    "axes.labelcolor": "black",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

# The five specified manuscript colors; black and white are neutral line/text colors.
CORAL = "#E38067"
INK = "#271A92"
LAVENDER = "#724BB6"
VIOLET = "#6F15AC"
MAGENTA = "#A730A2"
WHITE = "#FFFFFF"


def tint(hex_color: str, amount: float) -> str:
    """Blend a manuscript color with white to keep dense triangle meshes legible."""
    rgb = np.asarray([int(hex_color[i:i + 2], 16) for i in (1, 3, 5)], dtype=float)
    mixed = np.rint((1.0 - amount) * 255.0 + amount * rgb).astype(int)
    return "#" + "".join(f"{channel:02X}" for channel in mixed)


SURFACE_FILL = tint(LAVENDER, 0.12)
SURFACE_EDGE = tint(INK, 0.43)
CGAL_FILL = tint(VIOLET, 0.43)


@dataclass
class Case:
    label: str
    base_vertices: np.ndarray
    base_faces: np.ndarray
    output_vertices: np.ndarray
    patch_faces: np.ndarray
    loop: np.ndarray
    cue: np.ndarray | None = None
    output_faces: np.ndarray | None = None
    provenance: dict | None = None

    def all_output_faces(self) -> np.ndarray:
        if self.output_faces is not None:
            return self.output_faces
        return np.vstack([self.base_faces, self.patch_faces])


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def checked_arrays(path: Path, need: tuple[str, ...]) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as item:
        missing = set(need) - set(item.files)
        if missing:
            raise ValueError(f"Missing {sorted(missing)} in {path}")
        return {key: np.asarray(item[key]) for key in item.files}


def checked_case(case: Case) -> Case:
    if len(case.loop) < 3 or len(case.patch_faces) == 0:
        raise ValueError(f"No accepted patch and loop for {case.label}")
    if case.base_faces.max() >= len(case.base_vertices):
        raise ValueError(f"Invalid input faces: {case.label}")
    if case.patch_faces.max() >= len(case.output_vertices):
        raise ValueError(f"Invalid patch indices: {case.label}")
    if case.output_faces is not None and case.output_faces.max() >= len(case.output_vertices):
        raise ValueError(f"Invalid output faces: {case.label}")
    return case


def from_policy(suite: str, index: int, label: str) -> Case:
    # Exactly the frozen prior-locked joint-pool prediction, read from saved files.
    directory = ROOT / "experiments" / suite
    protocol = load_json(directory / "protocol.json")
    item = protocol["cases"][index]
    prediction = load_json(directory / "predictions" / f"{index:03d}.json")
    choice = prediction["decisions"]["prior_locked_pool_with_joint"]
    if choice["decision"] != "accept":
        raise ValueError(f"Case {suite}/{index} abstained")
    selected = choice["selected_loop_id"]
    base = checked_arrays(directory / "work" / f"case{index:03d}" / f"loop{selected}" / "damaged_input.npz", ("vertices", "faces", "selected_loop"))
    result = checked_arrays(Path(choice["patch_path"]), ("vertices", "patch_faces"))
    loop = np.asarray(base["selected_loop"], dtype=int)
    if not np.array_equal(np.sort(loop), np.sort(np.asarray(item["target_loop_vertices"], dtype=int))):
        raise ValueError(f"Selected rim differs from target in {suite}/{index}")
    return checked_case(Case(label, base["vertices"], base["faces"], result["vertices"], result["patch_faces"], loop, np.asarray(item.get("cue", []), dtype=float)))


def added_faces(base_faces: np.ndarray, output_faces: np.ndarray) -> np.ndarray:
    source = {tuple(sorted(map(int, face))) for face in np.asarray(base_faces)}
    added = [face for face in output_faces if tuple(sorted(map(int, face))) not in source]
    return np.asarray(added, dtype=np.int64).reshape(-1, 3)


def from_scan_edit(name: str, label: str) -> Case:
    base = checked_arrays(ROOT / "data" / "stanford_scan_edits" / "cases" / f"{name}.npz", ("vertices", "faces"))
    output = checked_arrays(ROOT / "experiments" / "scan_edit_benchmark" / "outputs" / name / "tcmr.npz", ("vertices", "faces"))
    labels = load_json(ROOT / "experiments" / "scan_edit_benchmark" / "score_labels.json")
    item = next(row for row in labels["rows"] if row["sample_id"] == name)
    original_faces = {tuple(sorted(map(int, face))) for face in base["faces"]}
    repaired_faces = {tuple(sorted(map(int, face))) for face in output["faces"]}
    if not original_faces.issubset(repaired_faces):
        raise ValueError(f"Saved scan-edit output reindexed or changed retained faces: {name}")
    patch = added_faces(base["faces"], output["faces"])
    return checked_case(Case(label, base["vertices"], base["faces"], output["vertices"], patch, np.asarray(item["target_loop_vertices"], dtype=int), base.get("cue"), output["faces"]))


def from_natural(name: str, label: str) -> Case:
    directory = ROOT / "experiments" / "real_scan_external" / "cases" / name
    original = checked_arrays(directory / "input.npz", ("vertices", "faces", "target_loop"))
    inference = load_json(directory / "inference" / "inference.json")
    choice = inference["decisions"]["prior_locked_pool_with_joint"]
    if choice["decision"] != "accept":
        raise ValueError(f"Natural scan {name} abstained")
    selected = choice["selected_loop_id"]
    base = checked_arrays(directory / "inference" / f"loop{selected}" / "damaged_input.npz", ("vertices", "faces", "selected_loop"))
    output = checked_arrays(Path(choice["patch_path"]), ("vertices", "patch_faces"))
    if not np.array_equal(base["vertices"], original["vertices"]) or not np.array_equal(base["faces"], original["faces"]):
        raise ValueError(f"Natural scan inference input changed: {name}")
    loop = np.asarray(base["selected_loop"], dtype=int)
    if set(loop) != set(np.asarray(original["target_loop"], dtype=int)):
        raise ValueError(f"Selected rim differs from designated natural scan hole: {name}")
    return checked_case(Case(label, base["vertices"], base["faces"], output["vertices"], output["patch_faces"], loop, original.get("cue")))


def from_robustness(sample_id: str, label: str) -> Case:
    directory = ROOT / "experiments" / "scan_robustness"
    protocol = load_json(directory / "protocol.json")
    item = next(row for row in protocol["cases"] if row["sample_id"] == sample_id)
    index = item["index"]
    prediction = load_json(directory / "predictions" / f"{index:03d}_tcmr.json")
    choice = prediction["decisions"]["prior_locked_pool_with_joint"]
    if choice["decision"] != "accept":
        raise ValueError(f"Robustness case {sample_id} abstained")
    base = checked_arrays(Path(choice["output_path"]).parent / "damaged_input.npz", ("vertices", "faces", "selected_loop"))
    result = checked_arrays(Path(choice["patch_path"]), ("vertices", "patch_faces"))
    return checked_case(Case(label, base["vertices"], base["faces"], result["vertices"], result["patch_faces"], base["selected_loop"].astype(int), np.asarray(item.get("cue", []), dtype=float)))


def to_poly(vertices: np.ndarray, faces: np.ndarray) -> pv.PolyData:
    tri = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    cells = np.column_stack((np.full(len(tri), 3, dtype=np.int64), tri)).ravel()
    return pv.PolyData(np.asarray(vertices, dtype=float), cells)


def local_faces(vertices: np.ndarray, faces: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    # Display crop only: it never changes a saved mesh or evaluation output.
    tri_center = vertices[np.asarray(faces, dtype=int)].mean(axis=1)
    selected = np.linalg.norm(tri_center - center, axis=1) < radius * 3.2
    return faces[selected] if selected.sum() >= 3 else faces


def view_vectors(case: Case) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    rim = case.base_vertices[case.loop]
    center = rim.mean(axis=0)
    centered = rim - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1].astype(float)
    body_center = case.base_vertices.mean(axis=0)
    if np.dot(normal, center - body_center) < 0:
        normal *= -1
    tangent = vh[0].astype(float)
    sight = normal + 0.25 * tangent
    sight /= np.linalg.norm(sight)
    up_hint = np.asarray([0.0, 0.0, 1.0])
    if abs(np.dot(up_hint, sight)) > 0.88:
        up_hint = np.asarray([0.0, 1.0, 0.0])
    right = np.cross(up_hint, sight)
    right /= np.linalg.norm(right)
    up = np.cross(sight, right)
    up /= np.linalg.norm(up)
    radius = max(float(np.linalg.norm(centered, axis=1).max()), 1e-4)
    return center, sight, up, radius


def add_rim(plotter: pv.Plotter, points: np.ndarray, color: str, width: float) -> None:
    if len(points) < 3:
        return
    line = pv.lines_from_points(np.vstack([points, points[0]]))
    plotter.add_mesh(line, color=color, line_width=width, render_lines_as_tubes=True, lighting=False)


def set_camera(plotter: pv.Plotter, case: Case, local: bool, oblique: bool = False) -> None:
    center, direction, up, radius = view_vectors(case)
    if oblique:
        direction = direction + 0.44 * up
        direction /= np.linalg.norm(direction)
    span = max(float(np.linalg.norm(np.ptp(case.base_vertices, axis=0))), radius * 3)
    focal = center if local else case.base_vertices.mean(axis=0) * 0.76 + center * 0.24
    distance = 5.5 * (radius if local else span)
    plotter.camera.position = tuple(focal + direction * distance)
    plotter.camera.focal_point = tuple(focal)
    plotter.camera.up = tuple(up)
    plotter.camera.parallel_projection = True
    if local:
        plotter.camera.parallel_scale = radius * 2.4
    else:
        vertices = case.base_vertices
        cam_right = np.cross(up, direction)
        cam_right /= np.linalg.norm(cam_right)
        projected_x = vertices @ cam_right
        projected_y = vertices @ up
        aspect = plotter.window_size[0] / plotter.window_size[1]
        plotter.camera.parallel_scale = max(np.ptp(projected_y) * 0.62, np.ptp(projected_x) / aspect * 0.62, radius * 2.0)


def render(case: Case, mode: str, width: int = 690, height: int = 620) -> np.ndarray:
    """Modes: input, input_detail, repair, detail, oblique, cgal, meshfix, ours_baseline."""
    local = mode in {"input_detail", "detail", "oblique", "cgal", "meshfix", "ours_baseline"}
    plotter = pv.Plotter(off_screen=True, window_size=(width, height))
    plotter.set_background(WHITE)
    plotter.enable_anti_aliasing("ssaa")
    center, _, _, radius = view_vectors(case)
    is_input = mode in {"input", "input_detail"}
    if is_input:
        vertices, faces = case.base_vertices, case.base_faces
    else:
        vertices, faces = case.output_vertices, case.all_output_faces()
        if len(case.patch_faces):
            patch_keys = {tuple(sorted(map(int, face))) for face in case.patch_faces}
            faces = np.asarray([face for face in faces if tuple(sorted(map(int, face))) not in patch_keys], dtype=np.int64).reshape(-1, 3)
    if local:
        faces = local_faces(vertices, faces, center, radius)
    if len(faces):
        plotter.add_mesh(
            to_poly(vertices, faces), color=SURFACE_FILL, opacity=1.0,
            show_edges=True, edge_color=SURFACE_EDGE, line_width=0.75 if local else 0.52,
            smooth_shading=False, ambient=0.82, diffuse=0.18, specular=0.0,
        )
    if not is_input and len(case.patch_faces):
        patch = case.patch_faces
        if local:
            patch = local_faces(case.output_vertices, patch, center, radius)
        plotter.add_mesh(
            to_poly(case.output_vertices, patch),
            color=CGAL_FILL if mode == "cgal" else CORAL,
            opacity=1.0, show_edges=True,
            edge_color=VIOLET if mode == "cgal" else INK,
            line_width=1.0 if local else 0.85,
            smooth_shading=False, ambient=0.78, diffuse=0.22, specular=0.0,
        )
    if mode != "meshfix":
        loop_points = case.output_vertices[case.loop] if not is_input and case.loop.max() < len(case.output_vertices) else case.base_vertices[case.loop]
        add_rim(plotter, loop_points, MAGENTA, width=2.5 if local else 2.0)
    if is_input and case.cue is not None and len(case.cue) >= 3:
        cue = np.asarray(case.cue[:3], dtype=float)
        span = max(float(np.linalg.norm(np.ptp(case.base_vertices, axis=0))), radius)
        plotter.add_mesh(pv.Sphere(radius=span * 0.012, center=cue), color=CORAL, smooth_shading=True)
    set_camera(plotter, case, local=local, oblique=mode == "oblique")
    image = plotter.screenshot(return_img=True)
    plotter.close()
    return image


def save_plate(name: str, rows: list[tuple[str, list[np.ndarray]]], headers: list[str], size: tuple[float, float]) -> None:
    nrows, ncols = len(rows), len(headers)
    fig, axes = plt.subplots(nrows, ncols, figsize=size, squeeze=False)
    plt.subplots_adjust(left=0.13, right=0.99, top=0.925, bottom=0.028, wspace=0.025, hspace=0.095)
    for col, header in enumerate(headers):
        box = axes[0, col].get_position()
        fig.text((box.x0 + box.x1) / 2, 0.966, header, ha="center", va="center", fontsize=11, color="black")
    for row, (row_label, images) in enumerate(rows):
        if len(images) != ncols:
            raise ValueError(f"Expected {ncols} images in {row_label}")
        for col, im in enumerate(images):
            ax = axes[row, col]
            ax.imshow(im, interpolation="lanczos")
            ax.axis("off")
        box = axes[row, 0].get_position()
        fig.text(0.066, (box.y0 + box.y1) / 2, row_label, rotation=90, ha="center", va="center", fontsize=11, color="black")
    PAPER.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER / f"{name}.pdf", bbox_inches="tight", pad_inches=0.035)
    if WRITE_PREVIEWS:
        PREVIEW.mkdir(parents=True, exist_ok=True)
        fig.savefig(PREVIEW / f"{name}.png", dpi=185, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)
    print(f"{name}: {PAPER / (name + '.pdf')}", flush=True)


def plate_external() -> None:
    items = [(3, "Candelabra · 004"), (23, "Four-legged · 024"), (31, "Goblet · 032"), (39, "Guitar · 040")]
    rows = []
    for index, label in items:
        case = from_policy("formal_coseg_external_benchmark", index, label)
        rows.append((label, [render(case, "input"), render(case, "repair"), render(case, "detail")]))
    headers = ["(a) Damaged", "(b) Repaired", "(c) Patch detail"]
    save_plate("qualitative_external_categories_1", rows[:2], headers, (7.45, 4.75))
    save_plate("qualitative_external_categories_2", rows[2:], headers, (7.45, 4.75))


def plate_cross_category() -> None:
    items = [(3, "Chair · 004"), (24, "Vase · 025"), (41, "Alien · 042")]
    rows = []
    for index, label in items:
        case = from_policy("formal_coseg_cross_category", index, label)
        rows.append((label, [render(case, "input"), render(case, "repair"), render(case, "detail")]))
    save_plate("qualitative_cross_category", rows, ["(a) Damaged", "(b) Repaired", "(c) Patch detail"], (7.45, 7.4))


def plate_scan_edits() -> None:
    items = [("bunny_cut1", "Bunny"), ("armadillo_cut1", "Armadillo"), ("drill_cut1", "Drill"), ("dragon_cut1", "Dragon")]
    cases = [from_scan_edit(name, label) for name, label in items]
    rows = [
        ("Damaged", [render(case, "input", 650, 580) for case in cases]),
        ("Repaired", [render(case, "repair", 650, 580) for case in cases]),
        ("Patch detail", [render(case, "detail", 650, 580) for case in cases]),
    ]
    save_plate("qualitative_scan_edits", rows, ["(a) Bunny", "(b) Armadillo", "(c) Drill", "(d) Dragon"], (7.6, 6.1))


def plate_natural_scan() -> None:
    items = [("bunny", "Bunny"), ("camel", "Camel"), ("femur", "Femur")]
    rows = []
    for name, label in items:
        case = from_natural(name, label)
        rows.append((label, [render(case, "input"), render(case, "repair"), render(case, "detail")]))
    save_plate("qualitative_natural_scans", rows, ["(a) Native hole", "(b) Repaired", "(c) Patch detail"], (7.45, 7.4))


def plate_perturbations() -> None:
    items = [("011_banana", "Banana"), ("037_scissors", "Scissors")]
    conditions = [("clean", "(a) Clean"), ("noise_high", "(b) Noisy"), ("resolution_quarter", "(c) Coarse"), ("hole_large", "(d) Large hole")]
    rows = []
    for prefix, label in items:
        cases = [from_robustness(f"{prefix}_{condition}", label) for condition, _ in conditions]
        rows.append((label, [render(case, "detail", 650, 580) for case in cases]))
    save_plate("qualitative_scan_perturbations", rows, [label for _, label in conditions], (7.6, 4.2))


def runtime_case(index: int, method: str, label: str) -> Case:
    # Independent holdout runtime outputs for CGAL, MeshFix, and the frozen method.
    directory = ROOT / "experiments" / "formal_coseg_external_holdout"
    item = load_json(directory / "protocol.json")["cases"][index]
    case_dir = directory / "runtime" / f"case{index:03d}"
    base = checked_arrays(case_dir / "cgal_pool" / "damaged_input.npz", ("vertices", "faces"))
    result = checked_arrays(case_dir / method / "returned_mesh.npz", ("vertices", "faces"))
    if method in {"cgal_pool", "tcmr"}:
        if not np.array_equal(result["before_vertices"], base["vertices"]):
            raise ValueError(f"Baseline comparison input mismatch: case{index:03d}/{method}")
        if set(map(int, result["selected_loop"])) != set(map(int, item["target_loop_vertices"])):
            raise ValueError(f"Baseline comparison target mismatch: case{index:03d}/{method}")
    patch = result.get("patch_faces")
    if patch is None:
        patch = np.empty((0, 3), dtype=np.int64)
    return Case(label, base["vertices"], base["faces"], result["vertices"], patch, np.asarray(item["target_loop_vertices"], dtype=int), np.asarray(item["cue"], dtype=float), result["faces"])


def plate_baselines() -> None:
    rows = []
    for index in [0, 20]:
        ours = runtime_case(index, "tcmr", f"Vase · {index+1:03d}")
        cgal = runtime_case(index, "cgal_pool", ours.label)
        meshfix = runtime_case(index, "meshfix", ours.label)
        # All panels share the target-centered world camera.  MeshFix's global
        # reindexing prevents marking only the added triangles faithfully.
        images = [render(ours, "input_detail"), render(cgal, "cgal"), render(meshfix, "meshfix"), render(ours, "ours_baseline")]
        rows.append((ours.label, images))
    save_plate("qualitative_fair_baselines", rows, ["(a) Damaged", "(b) CGAL", "(c) MeshFix", "(d) TCMR"], (7.6, 4.6))


def plate_patch_atlas() -> None:
    items = [
        ("formal_coseg_external_benchmark", 3, "Candelabra"),
        ("formal_coseg_external_benchmark", 17, "Four-legged"),
        ("formal_coseg_external_benchmark", 30, "Goblet"),
        ("formal_coseg_external_benchmark", 41, "Guitar"),
        ("formal_coseg_cross_category", 6, "Chair"),
        ("formal_coseg_cross_category", 45, "Alien"),
    ]
    cases = [from_policy(suite, index, label) for suite, index, label in items]
    fig, axes = plt.subplots(2, 3, figsize=(7.45, 4.5), squeeze=False)
    plt.subplots_adjust(left=0.02, right=0.99, top=0.95, bottom=0.025, wspace=0.025, hspace=0.18)
    for index, case in enumerate(cases):
        row, col = divmod(index, 3)
        ax = axes[row, col]
        ax.imshow(render(case, "oblique", 690, 620), interpolation="lanczos")
        ax.axis("off")
        box = ax.get_position()
        fig.text((box.x0 + box.x1) / 2, min(box.y1 + 0.018, 0.985), f"({chr(ord('a')+index)}) {case.label}", ha="center", va="center", fontsize=11, color="black")
    PAPER.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER / "qualitative_patch_atlas.pdf", bbox_inches="tight", pad_inches=0.035)
    if WRITE_PREVIEWS:
        PREVIEW.mkdir(parents=True, exist_ok=True)
        fig.savefig(PREVIEW / "qualitative_patch_atlas.png", dpi=185, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)
    print(f"qualitative_patch_atlas: {PAPER / 'qualitative_patch_atlas.pdf'}", flush=True)


def seam_edge_angles(case: Case) -> np.ndarray:
    """Absolute oriented seam angles, matching the unified evaluator exactly."""
    vertices = case.output_vertices

    def normals_and_edges(faces):
        triangles = vertices[faces]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        lengths = np.linalg.norm(normals, axis=1)
        normals /= np.maximum(lengths[:, None], 1e-12)
        lookup = {}
        for fi, face in enumerate(faces):
            for a, b in zip(face, np.roll(face, -1)):
                lookup.setdefault(tuple(sorted((int(a), int(b)))), []).append(fi)
        return normals, lookup

    retained_normals, retained_edges = normals_and_edges(case.base_faces)
    patch_normals, patch_edges = normals_and_edges(case.patch_faces)
    values = []
    for a, b in zip(case.loop, np.roll(case.loop, -1)):
        key = tuple(sorted((int(a), int(b))))
        retained = retained_edges.get(key, [])
        inserted = patch_edges.get(key, [])
        if len(retained) != 1 or len(inserted) != 1:
            raise ValueError(f"Non-unique seam adjacency for {case.label}: {key}")
        def direction(face):
            for x, y in zip(face, np.roll(face, -1)):
                if int(x) == int(a) and int(y) == int(b):
                    return 1
                if int(x) == int(b) and int(y) == int(a):
                    return -1
            raise ValueError("Missing oriented edge")
        nr = retained_normals[retained[0]] * direction(case.base_faces[retained[0]])
        npatch = patch_normals[inserted[0]] * (-direction(case.patch_faces[inserted[0]]))
        tangent = vertices[int(b)] - vertices[int(a)]
        tangent /= np.linalg.norm(tangent)
        values.append(abs(float(np.arctan2(np.cross(nr, npatch) @ tangent, nr @ npatch))))
    return np.asarray(values)


SEAM_CMAP = LinearSegmentedColormap.from_list(
    "interface_angle", [INK, LAVENDER, MAGENTA, CORAL], N=256
)


def mechanism_camera(case: Case, oblique_amount: float = 0.85) -> dict:
    """A shared oblique camera derived solely from the common initialization."""
    center, direction, up, radius = view_vectors(case)
    direction = direction + oblique_amount * up
    direction /= np.linalg.norm(direction)
    right = np.cross(up, direction)
    right /= np.linalg.norm(right)
    up = np.cross(direction, right)
    return {"center": center, "direction": direction, "up": up, "right": right,
            "radius": radius, "position": center + direction * radius * 12,
            "focal_point": center, "parallel_projection": True,
            "parallel_scale": radius * 1.43}


def mechanism_lighting(plotter: pv.Plotter, camera: dict) -> None:
    center, direction, up, right, radius = (camera[key] for key in ("center", "direction", "up", "right", "radius"))
    plotter.remove_all_lights()
    for offset, intensity in ((direction, 0.65),
                              (direction + 0.8 * up - 0.7 * right, 0.75),
                              (direction - 0.3 * up + 1.2 * right, 0.25)):
        plotter.add_light(pv.Light(position=tuple(center + radius * 8 * offset),
                                   focal_point=tuple(center), intensity=intensity,
                                   positional=False))


def section_specification(case: Case) -> dict:
    """Choose a readable edge using only the common initial geometry.

    The shorter of the two adjacent triangle cross-sections is maximized.
    Neither an output angle nor an improvement enters this selection.
    """
    candidates = []
    vertices = case.output_vertices
    for index, (a, b) in enumerate(zip(case.loop, np.roll(case.loop, -1))):
        retained = next(face for face in case.base_faces if a in face and b in face)
        patch = next(face for face in case.patch_faces if a in face and b in face)
        tangent = vertices[b] - vertices[a]
        length = np.linalg.norm(tangent)
        tangent /= length
        origin = (vertices[a] + vertices[b]) / 2
        lengths, radial = [], []
        for face in (retained, patch):
            third = next(vertex for vertex in face if vertex not in (a, b))
            delta = vertices[third] - origin
            along = delta @ tangent
            perpendicular = delta - along * tangent
            fraction = min(1.0, 0.5 / (0.5 + abs(along) / length))
            radial.append(perpendicular)
            lengths.append(float(np.linalg.norm(perpendicular * fraction)))
        candidates.append((min(lengths), index, int(a), int(b), retained, patch,
                           origin, tangent, radial[0], max(lengths)))
    chosen = max(candidates, key=lambda item: (item[0], -item[1]))
    _, index, a, b, retained, patch, origin, tangent, radial, extent = chosen
    x_axis = radial / np.linalg.norm(radial)
    y_axis = np.cross(tangent, x_axis)
    return {"edge_index": index, "edge_vertices": [a, b], "origin": origin,
            "plane_normal": tangent, "x_axis": x_axis, "y_axis": y_axis,
            "retained_face": retained, "patch_face": patch,
            "extent": float(extent), "minimum_section_length": float(chosen[0])}


def triangle_plane_section(vertices: np.ndarray, face: np.ndarray, specification: dict) -> np.ndarray:
    """Intersect an actual triangle with one fixed input-derived plane."""
    triangle = vertices[face]
    distances = (triangle - specification["origin"]) @ specification["plane_normal"]
    points = []
    for first in range(3):
        second = (first + 1) % 3
        d1, d2 = distances[first], distances[second]
        if abs(d1) < 1e-12:
            points.append(triangle[first])
        if d1 * d2 < 0:
            points.append(triangle[first] + (triangle[second] - triangle[first]) * d1 / (d1 - d2))
    unique = []
    for point in points:
        if not any(np.linalg.norm(point - prior) < 1e-10 for prior in unique):
            unique.append(point)
    if len(unique) != 2:
        raise ValueError("The shared input plane does not cut the displayed triangle uniquely")
    relative = np.asarray(unique) - specification["origin"]
    return np.column_stack((relative @ specification["x_axis"], relative @ specification["y_axis"]))


def section_edge_point(case: Case, specification: dict) -> np.ndarray:
    """Actual intersection of the marked seam edge with the fixed section plane."""
    first, second = case.output_vertices[np.asarray(specification["edge_vertices"])]
    direction = second - first
    fraction = -np.dot(first - specification["origin"], specification["plane_normal"]) / np.dot(direction, specification["plane_normal"])
    return first + fraction * direction


def section_panel(fig, box: list[float], case: Case, specification: dict, angle: float) -> None:
    """An undistorted, equally scaled section, not an exaggerated displacement."""
    ax = fig.add_axes([box[0], box[1], box[2] * 0.62, box[3]])
    extent = specification["extent"]
    for face, color in ((specification["retained_face"], LAVENDER),
                        (specification["patch_face"], CORAL)):
        section = triangle_plane_section(case.output_vertices, face, specification)
        ax.plot(section[:, 0], section[:, 1], color=color, lw=2.2, solid_capstyle="round")
    relative = section_edge_point(case, specification) - specification["origin"]
    ax.plot(relative @ specification["x_axis"], relative @ specification["y_axis"],
            "o", ms=3.4, color=INK, markeredgecolor=WHITE, markeredgewidth=0.6, zorder=5)
    ax.set_xlim(-extent * 1.12, extent * 1.12)
    ax.set_ylim(-extent * 0.55, extent * 0.55)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.text(box[0] + box[2] * 0.73, box[1] + box[3] * 0.53,
             rf"$\theta_e$ {angle:.2f}", ha="center", va="center", fontsize=12)


def context_camera(case: Case, width: int = 480, height: int = 820) -> dict:
    """A separate upright PCA view for locating the repair on the whole input."""
    vertices = case.base_vertices
    eigenvalues, axes = np.linalg.eigh(np.cov(vertices.T))
    up = axes[:, -1]
    if up[np.argmax(np.abs(up))] < 0:
        up = -up
    horizontal = axes[:, -2]
    if horizontal[np.argmax(np.abs(horizontal))] < 0:
        horizontal = -horizontal
    other = np.cross(up, horizontal)
    center = (vertices.max(axis=0) + vertices.min(axis=0)) / 2
    target_direction = vertices[case.loop].mean(axis=0) - center
    direction = 0.85 * horizontal + 0.55 * other
    if direction @ target_direction < 0:
        direction = -direction
    direction += (0.28 if target_direction @ up >= 0 else -0.28) * up
    direction /= np.linalg.norm(direction)
    right = np.cross(up, direction)
    right /= np.linalg.norm(right)
    up = np.cross(direction, right)
    relative = vertices - center
    projected_x, projected_y = relative @ right, relative @ up
    x_extent, y_extent = np.ptp(projected_x), np.ptp(projected_y)
    center += right * (projected_x.max() + projected_x.min()) / 2
    center += up * (projected_y.max() + projected_y.min()) / 2
    scale = max(y_extent * 0.55, x_extent / (width / height) * 0.55)
    position = center + direction * max(x_extent, y_extent) * 8
    return {"center": center, "direction": direction, "up": up, "right": right,
            "radius": float(max(x_extent, y_extent)), "parallel_scale": float(scale),
            "position": position, "focal_point": center, "parallel_projection": True,
            "pca_eigenvalues": eigenvalues, "orientation_rule": "Largest principal axis upright; side view from two transverse axes, tilted toward the input target rim"}


def render_context(case: Case, width: int = 480, height: int = 820) -> np.ndarray:
    """Show the entire damaged input, with its target rim marked in coral."""
    plotter = pv.Plotter(off_screen=True, window_size=(width, height))
    plotter.set_background(WHITE)
    plotter.enable_anti_aliasing("ssaa")
    camera = context_camera(case, width, height)
    mechanism_lighting(plotter, camera)
    plotter.add_mesh(to_poly(case.base_vertices, case.base_faces),
                     color=tint(LAVENDER, 0.29), smooth_shading=True,
                     show_edges=True, edge_color=tint(INK, 0.44), line_width=0.65,
                     ambient=0.45, diffuse=0.55, specular=0.10, specular_power=24)
    add_rim(plotter, case.base_vertices[case.loop], CORAL, 7.0)
    plotter.camera.position = tuple(camera["position"])
    plotter.camera.focal_point = tuple(camera["center"])
    plotter.camera.up = tuple(camera["up"])
    plotter.camera.parallel_projection = True
    plotter.camera.parallel_scale = camera["parallel_scale"]
    raster = plotter.screenshot(return_img=True)
    plotter.close()
    return raster


def render_mechanism(case: Case, angles: np.ndarray, vmax: float, width: int = 1050, height: int = 760,
                     camera_case: Case | None = None, crop_patch: bool = False,
                     oblique_amount: float = 0.85, seam_width: float = 12.0,
                     section: dict | None = None) -> np.ndarray:
    """Render the complete retained surface through a shared local viewport."""
    plotter = pv.Plotter(off_screen=True, window_size=(width, height))
    plotter.set_background(WHITE)
    plotter.enable_anti_aliasing("ssaa")
    camera = mechanism_camera(case if camera_case is None else camera_case, oblique_amount)
    center, direction, up, radius = (camera[key] for key in ("center", "direction", "up", "radius"))
    mechanism_lighting(plotter, camera)
    vertices = case.output_vertices
    plotter.add_mesh(
        to_poly(vertices, case.base_faces),
        color=tint(LAVENDER, 0.29), show_edges=True,
        edge_color=tint(INK, 0.55), line_width=0.92,
        smooth_shading=True, ambient=0.45, diffuse=0.55, specular=0.10, specular_power=24,
    )
    patch = case.patch_faces
    if crop_patch:
        patch = patch[np.linalg.norm(vertices[patch].mean(axis=1) - center, axis=1) < radius * 2.15]
    plotter.add_mesh(
        to_poly(vertices, patch),
        color=tint(CORAL, 0.74), show_edges=True,
        edge_color=tint(INK, 0.75), line_width=1.15,
        smooth_shading=False, ambient=0.45, diffuse=0.55, specular=0.10, specular_power=24,
    )
    segments = pv.PolyData()
    segments.points = vertices
    segments.lines = np.column_stack((np.full(len(case.loop), 2), case.loop, np.roll(case.loop, -1))).ravel()
    segments.cell_data["Seam angle"] = angles
    seam_actor = plotter.add_mesh(
        segments, scalars="Seam angle", cmap=SEAM_CMAP, clim=(0.0, vmax),
        show_scalar_bar=False, preference="cell", line_width=seam_width,
        render_lines_as_tubes=False, lighting=False,
    )
    # Resolve coincident raster depth without moving any edge or surface vertex.
    seam_actor.mapper.SetResolveCoincidentTopologyToPolygonOffset()
    seam_actor.mapper.SetRelativeCoincidentTopologyLineOffsetParameters(0.0, -12.0)
    plotter.camera.position = tuple(center + direction * radius * 12)
    plotter.camera.focal_point = tuple(center)
    plotter.camera.up = tuple(up)
    plotter.camera.parallel_projection = True
    plotter.camera.parallel_scale = camera["parallel_scale"]
    image = plotter.screenshot(return_img=True)
    if section is not None:
        # A print-sized annotation at the exact projected edge-plane crossing. The
        # marker is an overlay; mesh coordinates and the shared camera stay fixed.
        from PIL import Image, ImageDraw
        point = section_edge_point(case, section)
        plotter.renderer.SetWorldPoint(*point, 1.0)
        plotter.renderer.WorldToDisplay()
        px, py, _ = plotter.renderer.GetDisplayPoint()
        py = image.shape[0] - 1 - py
        overlay = Image.fromarray(image)
        draw = ImageDraw.Draw(overlay)
        marker_radius = width * 0.011
        draw.ellipse((px - marker_radius, py - marker_radius,
                      px + marker_radius, py + marker_radius),
                     fill=INK, outline=WHITE, width=max(3, round(width * 0.003)))
        image = np.asarray(overlay)
    plotter.close()
    return image


def plate_collar_mechanism(examples: list[tuple[str, list[Case]]], preview: Path | None = None,
                           write_metadata: bool = True) -> None:
    """Compare identical initial topology under fixed and movable collars."""
    if len(examples) != 2:
        raise ValueError("The mechanism plate uses exactly two observed cases")
    width, height = 7.14, 5.80
    fig = plt.figure(figsize=(width, height))
    columns = [0.91, 3.01, 5.11]
    cell_width = 1.99
    headers = ["Initialization", "Fixed collar", "Movable collar"]
    angles_all = [[seam_edge_angles(case) for case in cases] for _, cases in examples]
    vmax = max(0.5, float(np.ceil(max(a.max() for group in angles_all for a in group) * 2) / 2))
    for x, header in zip(columns, headers):
        fig.text((x + cell_width / 2) / width, 5.56 / height, header, ha="center", va="center", fontsize=12)
    fig.text(0.39 / width, 5.56 / height, "Input", ha="center", va="center", fontsize=12)
    provenance = []
    for row, (label, cases) in enumerate(examples):
        if len(cases) != 3:
            raise ValueError("Expected CGAL / patch-only / joint outputs")
        baseline = cases[0]
        for candidate in cases[1:]:
            if not np.array_equal(candidate.base_vertices, baseline.base_vertices) or not np.array_equal(candidate.base_faces, baseline.base_faces):
                raise ValueError("The displayed input mesh differs across conditions")
            if not np.array_equal(candidate.patch_faces, baseline.patch_faces):
                raise ValueError("The displayed patch topology differs across conditions")
        y = 3.84 if row == 0 else 1.40
        fig.text(0.045 / width, (y + 1.40) / height, label, ha="left", va="center", fontsize=12, fontweight="bold")
        specification = section_specification(baseline)
        context_ax = fig.add_axes([0.025 / width, (y - 0.02) / height, 0.74 / width, 1.32 / height])
        context_ax.imshow(render_context(baseline), interpolation="lanczos")
        context_ax.axis("off")
        camera = mechanism_camera(baseline)
        provenance.append({**(baseline.provenance or {}),
                           "camera": {key: value.tolist() if isinstance(value, np.ndarray) else value
                                      for key, value in camera.items()},
                           "context_camera": {key: value.tolist() if isinstance(value, np.ndarray) else value
                                              for key, value in context_camera(baseline).items()},
                           "section": {key: value.tolist() if isinstance(value, np.ndarray) else value
                                       for key, value in specification.items()}})
        for col, (x, case) in enumerate(zip(columns, cases)):
            angles = angles_all[row][col]
            method = ("initializer", "patch_only", "joint")[col]
            provenance[-1]["methods"][method]["local_edge_angle_rad"] = float(angles[specification["edge_index"]])
            image = render_mechanism(case, angles, vmax, width=1380, height=900,
                                     camera_case=baseline, section=specification)
            ax = fig.add_axes([x / width, y / height, cell_width / width, 1.30 / height])
            ax.imshow(image, interpolation="lanczos")
            ax.axis("off")
            section_panel(fig, [x / width, (y - 0.44) / height, cell_width / width, 0.32 / height],
                          case, specification, float(angles[specification["edge_index"]]))
            fig.text((x + cell_width / 2) / width, (y - 0.65) / height,
                     f"Mean: {angles.mean():.3f} rad", ha="center", va="center", fontsize=12)
            print(f"{label}; {headers[col]}; mean={angles.mean():.9f}; n_edges={len(angles)}", flush=True)
    cax = fig.add_axes([4.58 / width, 0.30 / height, 2.35 / width, 0.105 / height])
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, vmax), cmap=SEAM_CMAP),
                            cax=cax, orientation="horizontal", ticks=[0, vmax / 2, vmax])
    colorbar.ax.set_xticklabels(["0", f"{vmax / 2:g}", f"{vmax:g}"])
    colorbar.ax.tick_params(labelsize=12, length=2.5, pad=3)
    colorbar.outline.set_linewidth(0.6)
    fig.text(0.06 / width, 0.35 / height, "Seam angle (rad)", ha="left", va="center", fontsize=12)
    fig.text(2.02 / width, 0.35 / height, "Local section", ha="left", va="center", fontsize=12)
    output = PAPER / "collar_mechanism_comparison.pdf"
    fig.savefig(output, metadata={"Title": "Fixed and movable collar comparison", "Creator": "Python / Matplotlib and PyVista"})
    if preview is not None:
        preview.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview, dpi=220)
    plt.close(fig)
    if not write_metadata:
        print(output, flush=True)
        return
    metadata_path = ROOT / "experiments/paper_figure_source_data.json"
    document = load_json(metadata_path) if metadata_path.exists() else {}
    document["collar_mechanism_comparison"] = {
        "output": str(output), "backend": "Python / Matplotlib and PyVista",
        "selection_rule": "Within each cohort and the three-way native-valid intersection, choose the source nearest the cohort median fixed-minus-movable mean seam effect; break ties by source index. The original selected sources are unchanged.",
        "image_integrity": "Saved mesh coordinates and topology unchanged. Complete retained mesh rendered through shared input-derived oblique parallel cameras. Sections intersect the same fixed input plane with actual adjacent triangles; no anisotropic scaling or displacement exaggeration.",
        "section_selection_rule": "Using only the common initializer, maximize the shorter actual cross-section length of the two seam-adjacent triangles. The plane passes through the input edge midpoint and is normal to the input edge. No output effect enters selection.",
        "seam_color_limits_rad": [0.0, vmax], "font": "Helvetica", "font_size_pt": 12,
        "render_style": {"retained_color": tint(LAVENDER, 0.29), "patch_color": tint(CORAL, 0.74),
                         "retained_edge_color": tint(INK, 0.55), "patch_edge_color": tint(INK, 0.75),
                         "ambient": 0.45, "diffuse": 0.55, "specular": 0.10,
                         "retained_smooth_shading": True, "patch_smooth_shading": False,
                         "lights": "Shared camera-relative frontal fill, oblique key, and opposite fill"},
        "figure_dimensions_inches": [width, height], "examples": provenance,
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    metadata_path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(output, flush=True)


def collar_examples(results_path: Path) -> list[tuple[str, list[Case]]]:
    """One common-valid case nearest the median collar benefit per cohort."""
    results = load_json(results_path)
    if not results.get("complete", False):
        raise ValueError("Representative cases are selected only from the completed experiment")
    examples = []
    for dataset, title in [("holdout50", "a  Independent holdout"), ("cross_category60", "b  Cross-category test")]:
        rows = [row for row in results["rows"] if row["dataset"] == dataset
                and all(row["methods"].get(method, {}).get("valid", False) for method in ("initializer", "patch_only", "joint"))]
        if not rows:
            raise ValueError(f"No common-valid triples in {dataset}")
        def gain(row):
            methods = row["methods"]
            return methods["patch_only"]["geometry"]["seam_normal_absolute_mean"] - methods["joint"]["geometry"]["seam_normal_absolute_mean"]
        median_gain = float(np.median([gain(row) for row in rows]))
        selected = min(rows, key=lambda row: (abs(gain(row) - median_gain), row["index"]))
        base = checked_arrays(Path(selected["input_path"]), ("vertices", "faces", "selected_loop"))
        cases = []
        for method in ("initializer", "patch_only", "joint"):
            entry = selected["methods"][method]
            output = checked_arrays(Path(entry["patch_path"]), ("vertices", "patch_faces"))
            case = checked_case(Case(title, base["vertices"], base["faces"], output["vertices"], output["patch_faces"], base["selected_loop"].astype(int)))
            expected = float(entry["geometry"]["seam_normal_absolute_mean"])
            if not np.isclose(seam_edge_angles(case).mean(), expected, atol=1e-10, rtol=1e-9):
                raise ValueError(f"Seam-angle calculation differs from unified evaluator: {dataset}/{selected['index']}/{method}")
            if method == "patch_only" and not np.array_equal(case.output_vertices[:len(case.base_vertices)], case.base_vertices):
                raise ValueError("The fixed-collar rendering changes retained vertices")
            cases.append(case)
        print(json.dumps({"dataset": dataset, "index": selected["index"], "source_model_id": selected["source_model_id"],
                          "common_valid_n": len(rows), "cohort_median_gain_rad": median_gain,
                          "displayed_gain_rad": gain(selected)}, ensure_ascii=False), flush=True)
        cases[0].provenance = {
            "dataset": dataset, "index": selected["index"], "source_model_id": selected["source_model_id"],
            "input_path": selected["input_path"],
            "input_sha256": hashlib.sha256(Path(selected["input_path"]).read_bytes()).hexdigest(),
            "common_valid_count": len(rows), "cohort_median_gain_rad": median_gain,
            "displayed_gain_rad": gain(selected),
            "methods": {method: {"patch_path": selected["methods"][method]["patch_path"],
                                  "patch_sha256": hashlib.sha256(Path(selected["methods"][method]["patch_path"]).read_bytes()).hexdigest(),
                                  "mean_seam_rad": float(seam_edge_angles(case).mean())}
                        for method, case in zip(("initializer", "patch_only", "joint"), cases)},
        }
        examples.append((title, cases))
    return examples


def plate_reference_interfaces(preview: Path | None = None) -> None:
    """Show a prespecified smooth and sharp interface with original geometry.

    The two sources are chosen by median reference-error change, without an
    improvement filter. All columns share the damaged-input camera and scale.
    The common angle map shows the measured original and reconstructed folds;
    per-panel error labels distinguish continuity from reference agreement.
    """
    from build_coseg_external_benchmark import read_off

    evidence = load_json(ROOT / "experiments/formal_collar_mechanism/reference_geometry.json")
    core = load_json(ROOT / "experiments/formal_collar_mechanism/results.json")
    records = {(r["dataset"], r["index"]): r for r in core["rows"]}
    rows = {(r["dataset"], r["index"]): r for r in evidence["rows"]}
    methods = ("patch_only", "joint", "bounded_biharmonic_fair")
    headers = ("Reference", "Fixed collar", "Joint collar", "Bounded fairing")
    width, height = 7.14, 4.65
    fig = plt.figure(figsize=(width, height))
    starts = np.array([0.04, 1.82, 3.60, 5.38])
    cell_width, cell_height = 1.72, 1.48
    vmax = np.pi / 2
    for x, header in zip(starts, headers):
        fig.text((x + cell_width / 2) / width, 4.48 / height, header,
                 ha="center", va="center", fontsize=11)
    for row_index, (stratum, label) in enumerate((
        ("smooth_interface", "a  Smooth interface"),
        ("sharp_interface", "b  Sharp interface"),
    )):
        selected = evidence["visualization_cases"][stratum]
        key = (selected["dataset"], selected["index"])
        record, evaluated = records[key], rows[key]
        base = checked_arrays(Path(record["input_path"]), ("vertices", "faces", "selected_loop"))
        base_v, base_f, loop = base["vertices"], base["faces"], base["selected_loop"].astype(int)
        cases, measured, errors = [], [], []
        for method in methods:
            entry = evaluated["methods"][method]
            if entry["status"] != "success":
                raise ValueError(f"No reference score for {key}/{method}")
            output = checked_arrays(Path(entry["patch_path"]), ("vertices", "patch_faces"))
            cases.append(checked_case(Case(label, base_v, base_f, output["vertices"], output["patch_faces"], loop)))
            measured.append(np.asarray(entry["reference_seam"]["output_angles_rad"]))
            errors.append(float(entry["reference_seam"]["mean_absolute_error_rad"]))
        full_v, full_f = read_off(Path(selected["raw_mesh"]))
        construction = next(item for item in load_json(Path(selected["construction_protocol"]))["cases"]
                            if item["sample_id"] == record["sample_id"])
        if "retained_original_vertex_ids" in construction:
            source_ids = np.asarray(construction["retained_original_vertex_ids"], dtype=int)
        else:
            segmentation = np.fromstring(Path(selected["raw_segmentation"]).read_text(encoding="utf-8"), sep=" ", dtype=int)
            keep = ~np.isin(segmentation, [construction["target_label"], construction["distractor_label"]])
            source_ids = np.unique(full_f[keep])
        if not np.array_equal(full_v[source_ids], base_v):
            raise ValueError("Original retained vertex identities do not reconstruct the input")
        original_loop = np.asarray(selected["original_reference_seam_edges"], dtype=int)[:, 0]
        if not np.array_equal(full_v[original_loop], base_v[loop]):
            raise ValueError("Original and damaged rim coordinates differ")
        reference = checked_case(Case(label, full_v, source_ids[base_f], full_v,
                                      full_f[np.asarray(selected["gt_target_source_face_ids"], dtype=int)], original_loop))
        ref_angles = np.asarray(evaluated["reference"]["seam_reference_angles_rad"])
        all_cases, all_angles = [reference] + cases, [ref_angles] + measured
        y = 2.64 if row_index == 0 else 0.65
        fig.text(0.09 / width, (y + 1.64) / height, label,
                 ha="left", va="center", fontsize=11, fontweight="bold")
        for column, (x, case, angles) in enumerate(zip(starts, all_cases, all_angles)):
            raster = render_mechanism(case, angles, vmax, width=860, height=740,
                                      camera_case=cases[0], crop_patch=True)
            ax = fig.add_axes([x / width, y / height, cell_width / width, cell_height / height])
            ax.imshow(raster, interpolation="lanczos")
            ax.axis("off")
            value = f"Angle: {ref_angles.mean():.3f} rad" if column == 0 else f"MAE: {errors[column - 1]:.3f} rad"
            fig.text((x + cell_width / 2) / width, (y - 0.13) / height, value,
                     ha="center", va="center", fontsize=10.8)
        print(json.dumps({"stratum": stratum, "dataset": key[0], "index": key[1],
                          "reference_mean_rad": float(ref_angles.mean()),
                          "reference_mae_rad": dict(zip(methods, errors))}), flush=True)
    cax = fig.add_axes([4.60 / width, 0.21 / height, 2.18 / width, 0.10 / height])
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, vmax), cmap=SEAM_CMAP),
                      cax=cax, orientation="horizontal", ticks=[0, np.pi / 4, np.pi / 2])
    bar.ax.set_xticklabels(["0", "0.79", "1.57"])
    bar.ax.tick_params(labelsize=10.7, length=2.5, pad=3)
    bar.outline.set_linewidth(0.6)
    fig.text(0.14 / width, 0.26 / height, "Boundary color: unoriented seam angle (rad)",
             ha="left", va="center", fontsize=10.8)
    output = PAPER / "reference_interface_comparison.pdf"
    fig.savefig(output, metadata={"Creator": "Python / Matplotlib and PyVista"})
    if preview is not None:
        preview.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview, dpi=220)
    plt.close(fig)
    print(output, flush=True)


def plate_attachment_comparison(preview: Path | None = None, write_metadata: bool = True) -> None:
    """Show median-effect examples without changing meshes or selecting by appearance.

    This image plate compares the seam/displacement tradeoff of saved, verified
    candidates from identical initial geometry. The common-valid restriction and
    median effect rule are fixed before any rendering. Original coordinates,
    a shared camera per input, and one boundary-angle scale preserve geometry.
    """
    plt.rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": "Helvetica",
                         "mathtext.it": "Helvetica:italic", "mathtext.bf": "Helvetica:bold"})
    source_paths = {
        "core": ROOT / "experiments/formal_collar_mechanism/results.json",
        "attachment": ROOT / "experiments/formal_movable_neighborhood/attachment_results.json",
        "bounded": ROOT / "experiments/formal_movable_neighborhood/bounded_fair_results.json",
    }
    sources = {name: load_json(path) for name, path in source_paths.items()}
    if not all(source.get("complete", False) for source in sources.values()):
        raise ValueError("The comparison requires completed frozen experiment results")
    lookup = {
        name: {(row["dataset"], row["index"]): row for row in source["rows"]}
        for name, source in sources.items()
    }
    methods = ("initializer", "calatroni_attachment", "bounded_biharmonic_fair", "joint")
    headers = ("Initialization", "Variational\nattachment", "Bounded\nfairing", "Bounded joint\nrefinement")
    selected_rows, example_cases, metadata = [], [], []
    for dataset, label in (("holdout50", "a  Independent holdout"),
                           ("cross_category60", "b  Cross-category test")):
        eligible = []
        for key, row in lookup["core"].items():
            if key[0] != dataset:
                continue
            entries = {method: row["methods"][method] for method in ("initializer", "joint")}
            entries.update(lookup["attachment"][key]["methods"])
            entries.update(lookup["bounded"][key]["methods"])
            if all(entries[method].get("valid", False) for method in methods):
                effect = (entries["joint"]["geometry"]["seam_normal_absolute_mean"]
                          - entries["calatroni_attachment"]["geometry"]["seam_normal_absolute_mean"])
                eligible.append((row, entries, float(effect)))
        median = float(np.median([item[2] for item in eligible]))
        row, entries, effect = min(eligible, key=lambda item: (abs(item[2] - median), item[0]["index"]))
        selected_rows.append((row, entries, label))
        base = checked_arrays(Path(row["input_path"]), ("vertices", "faces", "selected_loop"))
        cases, saved_metrics = [], {}
        for method in methods:
            entry = entries[method]
            output = checked_arrays(Path(entry["patch_path"]), ("vertices", "patch_faces"))
            case = checked_case(Case(label, base["vertices"], base["faces"], output["vertices"],
                                     output["patch_faces"], base["selected_loop"].astype(int)))
            if cases and not np.array_equal(case.patch_faces, cases[0].patch_faces):
                raise ValueError("Patch topology differs across comparison methods")
            geometry = entry["geometry"]
            measured = seam_edge_angles(case)
            if not np.isclose(measured.mean(), geometry["seam_normal_absolute_mean"], atol=1e-10, rtol=1e-9):
                raise ValueError("Rendered seam angles differ from the unified evaluator")
            delta = case.output_vertices[:len(base["vertices"])] - base["vertices"]
            rms = float(np.sqrt(np.square(delta).sum(axis=1).mean()) / geometry["median_input_edge"])
            if not np.isclose(rms, geometry["retained_displacement_rms_over_edge"], atol=1e-11, rtol=1e-9):
                raise ValueError("Rendered retained displacement differs from the unified evaluator")
            saved_metrics[method] = {
                "patch_path": entry["patch_path"],
                "patch_sha256": hashlib.sha256(Path(entry["patch_path"]).read_bytes()).hexdigest(),
                "mean_seam_rad": float(measured.mean()),
                "retained_rms_over_h_in": rms,
                "native_valid": True,
            }
            cases.append(case)
        camera = mechanism_camera(cases[0])
        specification = section_specification(cases[0])
        for method, case in zip(methods, cases):
            saved_metrics[method]["local_edge_angle_rad"] = float(seam_edge_angles(case)[specification["edge_index"]])
        metadata.append({
            "dataset": dataset, "index": row["index"], "sample_id": row["sample_id"],
            "source_model_id": row["source_model_id"], "input_path": row["input_path"],
            "input_sha256": hashlib.sha256(Path(row["input_path"]).read_bytes()).hexdigest(),
            "common_valid_count": len(eligible), "median_joint_minus_attachment_mean_seam_rad": median,
            "selected_joint_minus_attachment_mean_seam_rad": effect,
            "methods": saved_metrics,
            "camera": {key: value.tolist() if isinstance(value, np.ndarray) else value
                       for key, value in camera.items()},
            "context_camera": {key: value.tolist() if isinstance(value, np.ndarray) else value
                               for key, value in context_camera(cases[0]).items()},
            "section": {key: value.tolist() if isinstance(value, np.ndarray) else value
                        for key, value in specification.items()},
        })
        example_cases.append(cases)
    maximum_angle = max(float(seam_edge_angles(case).max()) for cases in example_cases for case in cases)
    vmax = max(0.5, float(np.ceil(maximum_angle * 4) / 4))
    width, height = 7.14, 5.80
    fig = plt.figure(figsize=(width, height))
    starts = np.linspace(0.91, 5.61, 4)
    cell_width, cell_height = 1.49, 1.30
    for x, header in zip(starts, headers):
        fig.text((x + cell_width / 2) / width, 5.56 / height, header,
                 ha="center", va="center", fontsize=12, linespacing=1.02)
    fig.text(0.39 / width, 5.56 / height, "Input", ha="center", va="center", fontsize=12)
    for row_index, ((row, entries, label), cases) in enumerate(zip(selected_rows, example_cases)):
        y = 3.84 if row_index == 0 else 1.40
        fig.text(0.045 / width, (y + 1.40) / height, label,
                 ha="left", va="center", fontsize=12, fontweight="bold")
        specification = section_specification(cases[0])
        context_ax = fig.add_axes([0.025 / width, (y - 0.02) / height, 0.74 / width, 1.32 / height])
        context_ax.imshow(render_context(cases[0]), interpolation="lanczos")
        context_ax.axis("off")
        for x, method, case in zip(starts, methods, cases):
            angles = seam_edge_angles(case)
            image = render_mechanism(case, angles, vmax, width=1030, height=900,
                                     camera_case=cases[0], crop_patch=False,
                                     seam_width=12.0, section=specification)
            ax = fig.add_axes([x / width, y / height, cell_width / width, cell_height / height])
            ax.imshow(image, interpolation="lanczos")
            ax.axis("off")
            metric = metadata[row_index]["methods"][method]
            section_panel(fig, [x / width, (y - 0.44) / height, cell_width / width, 0.32 / height],
                          case, specification, float(angles[specification["edge_index"]]))
            fig.text((x + cell_width / 2) / width, (y - 0.65) / height,
                     f"Mean: {metric['mean_seam_rad']:.3f}",
                     ha="center", va="center", fontsize=12)
            fig.text((x + cell_width / 2) / width, (y - 0.85) / height,
                     rf"RMS/$h_{{\mathrm{{in}}}}$: {metric['retained_rms_over_h_in']:.4f}",
                     ha="center", va="center", fontsize=12)
    cax = fig.add_axes([4.58 / width, 0.30 / height, 2.35 / width, 0.105 / height])
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, vmax), cmap=SEAM_CMAP),
                      cax=cax, orientation="horizontal", ticks=[0, vmax / 2, vmax])
    bar.ax.set_xticklabels(["0", f"{vmax / 2:g}", f"{vmax:g}"])
    bar.ax.tick_params(labelsize=12, length=2.5, pad=2)
    bar.outline.set_linewidth(0.6)
    fig.text(0.06 / width, 0.35 / height, "Seam angle (rad)",
             ha="left", va="center", fontsize=12)
    fig.text(2.02 / width, 0.35 / height, "Local section", ha="left", va="center", fontsize=12)
    output = PAPER / "attachment_comparison.pdf"
    fig.savefig(output, metadata={"Creator": "Python / Matplotlib and PyVista"})
    if preview is not None:
        preview.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview, dpi=240)
    plt.close(fig)
    if not write_metadata:
        print(output, flush=True)
        return
    metadata_path = ROOT / "experiments/paper_figure_source_data.json"
    document = load_json(metadata_path) if metadata_path.exists() else {}
    document["attachment_comparison"] = {
        "output": str(output), "backend": "Python / Matplotlib and PyVista",
        "selection_rule": "Within each cohort and the four-way native-valid intersection, choose the source whose joint-minus-attachment mean seam effect is closest to the cohort median; break ties by source index. Selection precedes rendering.",
        "source_results": {name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                           for name, path in source_paths.items()},
        "image_integrity": "Original stored vertices; no registration, geometry scaling, displacement exaggeration, optimization, or alternative-case selection. Complete retained mesh and entire patch rendered through one shared input-derived oblique parallel viewport per row. Sections use one fixed input plane and equal 2D scale.",
        "section_selection_rule": "Using only the common initializer, maximize the shorter actual cross-section length of the two seam-adjacent triangles. The plane passes through the input edge midpoint and is normal to the input edge. No output effect enters selection.",
        "seam_color_limits_rad": [0.0, vmax], "font": "Helvetica", "font_size_pt": 12,
        "render_style": {"retained_color": tint(LAVENDER, 0.29), "patch_color": tint(CORAL, 0.74),
                         "retained_edge_color": tint(INK, 0.55), "patch_edge_color": tint(INK, 0.75),
                         "ambient": 0.45, "diffuse": 0.55, "specular": 0.10,
                         "retained_smooth_shading": True, "patch_smooth_shading": False,
                         "lights": "Shared camera-relative frontal fill, oblique key, and opposite fill"},
        "figure_dimensions_inches": [width, height], "examples": metadata,
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    metadata_path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(document["attachment_comparison"], indent=2, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=("all", "collar", "reference", "attachment"), default="all")
    parser.add_argument("--collar-results", type=Path, default=ROOT / "experiments" / "formal_collar_mechanism" / "results.json")
    parser.add_argument("--preview", type=Path)
    parser.add_argument("--no-metadata", action="store_true",
                        help="Render selected plates without updating shared source metadata")
    args = parser.parse_args()
    if args.only == "attachment":
        plate_attachment_comparison(args.preview, write_metadata=not args.no_metadata)
        return
    if args.only == "reference":
        plate_reference_interfaces(args.preview)
        return
    if args.only == "collar":
        plate_collar_mechanism(collar_examples(args.collar_results), args.preview,
                               write_metadata=not args.no_metadata)
        return
    # A plate is emitted only after its inputs and selected output paths pass validation.
    plate_external()
    plate_cross_category()
    plate_scan_edits()
    plate_natural_scan()
    plate_perturbations()
    plate_baselines()
    plate_patch_atlas()


if __name__ == "__main__":
    main()
