"""Fixed, input-derived G0/G1 guidance for local patch refinement.

The 30 degree edge classification follows Calatroni et al., *A Unified
Surface Geometric Framework for Feature-Aware Denoising, Hole Filling and
Context-Aware Completion*, JMIV 65 (2023), Eq. (5) and the blending masks
in Section 5, https://doi.org/10.1007/s10851-022-01107-w .

This module supplies a fixed mask and initial normal-jump targets. It does
not implement that paper's variational model or Guided Mesh Normal
Filtering. All inputs are available at inference: the damaged mesh, its
boundary loop, and the patch initializer. In particular, a sharp seam in
the initializer can be an artificial fold; the missing side's true angle
cannot be identified uniquely from the retained side alone.
"""
from __future__ import annotations

import numpy as np


def _unit_normals(vertices, faces):
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0],
                     triangles[:, 2] - triangles[:, 0])
    length = np.linalg.norm(cross, axis=1)
    return cross / np.maximum(length[:, None], 1e-15), length > 1e-15


def _seam_pairs(retained_faces, patch_faces, loop):
    """Match the optimizer's loop-order pairing, including skipped edges."""
    retained_map = {}
    patch_map = {}
    for faces, lookup in ((retained_faces, retained_map), (patch_faces, patch_map)):
        for face_id, face in enumerate(faces):
            for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
                lookup.setdefault((min(int(a), int(b)), max(int(a), int(b))), face_id)
    retained, patch, positions = [], [], []
    for position, (a, b) in enumerate(zip(loop, np.roll(loop, -1))):
        edge = (min(int(a), int(b)), max(int(a), int(b)))
        if edge in retained_map and edge in patch_map:
            retained.append(retained_map[edge])
            patch.append(patch_map[edge])
            positions.append(position)
    return (np.asarray(retained, dtype=np.int64),
            np.asarray(patch, dtype=np.int64),
            np.asarray(positions, dtype=np.int64))


def build_seam_guidance(vertices, retained_faces, patch_faces, boundary_loop,
                        feature_angle_degrees=30.0):
    """Return fixed NumPy arrays in ``optimizer._seam_pairs`` order.

    For each seam edge ``i``, let ``p_i`` be its patch unit normal after the
    optimizer's fixed sign alignment and ``r_i`` its retained unit normal.
    Its residual is ``delta_i = p_i - r_i``. When the initializer's unoriented
    seam angle is at least 30 degrees, ``delta_target[i] = delta_i(initial)``;
    otherwise the target is zero. ``normal_weight`` is identically one.

    Minimize the weighted mean square of ``error = delta - delta_target``.
    The variation term uses ``roll(error, -1) - error`` and
    ``variation_weight[i]``. That weight is zero when consecutive retained
    normals form an unoriented angle of at least 30 degrees. If a loop
    edge has no matching face, coupling across the gap is also disabled.
    Weight normalization should use the full edge count, retaining the
    original objective scale; do not divide by the number of active edges.

    Returned targets and weights must remain fixed during optimization.
    ``summary`` contains only JSON-serializable values for experiment logs.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    retained_faces = np.asarray(retained_faces, dtype=np.int64).reshape(-1, 3)
    patch_faces = np.asarray(patch_faces, dtype=np.int64).reshape(-1, 3)
    loop = np.asarray(boundary_loop, dtype=np.int64).reshape(-1)
    angle_degrees = float(feature_angle_degrees)
    if not 0.0 < angle_degrees <= 90.0:
        raise ValueError("Unoriented feature threshold must lie in (0, 90] degrees")
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("Vertices must be finite with shape (n, 3)")
    sr, sp, positions = _seam_pairs(retained_faces, patch_faces, loop)
    retained_normals, retained_valid = _unit_normals(vertices, retained_faces)
    patch_normals, patch_valid = _unit_normals(vertices, patch_faces)
    rn = retained_normals[sr]
    pn = patch_normals[sp]
    valid = retained_valid[sr] & patch_valid[sp]
    dot = np.einsum("ij,ij->i", pn, rn)
    orientation = np.where(dot >= 0.0, 1.0, -1.0)
    seam_angle = np.arccos(np.clip(np.abs(dot), 0.0, 1.0))
    sharp_mask = (seam_angle >= np.deg2rad(angle_degrees)) & valid
    delta0 = orientation[:, None] * pn - rn
    delta_target = np.where(sharp_mask[:, None], delta0, 0.0)
    normal_weight = np.ones(len(sr), dtype=np.float64)
    variation_weight = np.ones(len(sr), dtype=np.float64)
    retained_angle = np.zeros(len(sr), dtype=np.float64)
    if len(sr):
        retained_dot = np.einsum("ij,ij->i", rn, np.roll(rn, -1, axis=0))
        retained_angle = np.arccos(np.clip(np.abs(retained_dot), 0.0, 1.0))
        adjacent = (np.roll(positions, -1) - positions) % len(loop) == 1
        variation_weight[(retained_angle >= np.deg2rad(angle_degrees)) | ~adjacent] = 0.0
        variation_weight[~valid | ~np.roll(valid, -1)] = 0.0
    if len(sr) < 2:
        variation_weight[:] = 0.0
    return {
        "seam_retained": sr,
        "seam_patch": sp,
        "loop_edge_positions": positions,
        "orientation": orientation,
        "normal_weight": normal_weight,
        "delta_target": delta_target,
        "variation_weight": variation_weight,
        "sharp_mask": sharp_mask,
        "initial_seam_angle_radians": seam_angle,
        "retained_transition_angle_radians": retained_angle,
        "summary": {
            "method": "fixed_input_G0_G1_mask",
            "feature_angle_degrees": angle_degrees,
            "seam_edges_n": int(len(sr)),
            "sharp_edges_n": int(sharp_mask.sum()),
            "sharp_edge_fraction": float(sharp_mask.mean()) if len(sr) else 0.0,
            "variation_cuts_n": int((variation_weight == 0.0).sum()),
            "variation_cut_fraction": float((variation_weight == 0.0).mean()) if len(sr) else 0.0,
            "degenerate_seam_edges_n": int((~valid).sum()),
            "skipped_loop_edges_n": int(len(loop) - len(sr)),
            "frozen_from_initializer": True,
            "ground_truth_used": False,
            "sharp_target": "initial_oriented_normal_jump",
            "smooth_target": "zero_normal_jump",
        },
    }


def build_observable_seam_guidance(vertices, retained_faces, patch_faces,
                                   boundary_loop, feature_angle_degrees=30.0,
                                   range_sigma=0.35):
    """Continue a crease only when both observed ends identify its other side.

    All feature decisions and target normals depend exclusively on the retained
    surface and observed boundary.  ``patch_faces`` is used for seam indexing;
    patch face normals and patch-only vertex coordinates are never inspected.
    At each endpoint of a seam edge, an observed interior sharp edge must extend
    outwards along its tangent.  Its two observed normal sheets must be reliable,
    one must match the retained seam face, and both endpoints must predict the
    same alternate sheet.  Unsupported or ambiguous edges retain zero jump.

    The normal-sheet estimator adapts the area mean and minimum consistency
    metric of Zhang et al., *Guided Mesh Normal Filtering*, CGF 34(7), 2015:
    https://github.com/bldeng/GuidedDenoising .  Its author-default range sigma
    is 0.35.  Vertex-neighbor patches are restricted to the connected observed
    normal sheet before choosing their minimum-consistency estimate.  The
    30-degree feature/continuation angle follows the feature classification of
    Calatroni et al., JMIV 2023.  The bilateral continuation and confidence
    gates are this adapter's geometric assumptions, not those papers' complete
    algorithms.  No absent sheet can be reconstructed when both-side evidence
    is missing; in particular, a planar rim does not by itself identify a cap.

    The output follows ``build_seam_guidance``'s array conventions, except that
    ``orientation`` is identically one because no patch normal is inspected.
    The optimizer retains its own initial patch-to-retained sign convention.
    If ``sharp_edges_n == 0``, bypassing the guidance objective preserves the
    old optimizer bit-for-bit.  All hyperparameters are frozen, not fitted.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    retained_faces = np.asarray(retained_faces, dtype=np.int64).reshape(-1, 3)
    patch_faces = np.asarray(patch_faces, dtype=np.int64).reshape(-1, 3)
    loop = np.asarray(boundary_loop, dtype=np.int64).reshape(-1)
    angle = float(feature_angle_degrees)
    sigma = float(range_sigma)
    if not 0.0 < angle <= 90.0 or not 0.0 < sigma < 1.0:
        raise ValueError("Expected angle in (0, 90] and range_sigma in (0, 1)")
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("Vertices must have shape (n, 3)")
    observed_ids = np.unique(retained_faces)
    if not np.isfinite(vertices[observed_ids]).all():
        raise ValueError("Retained coordinates must be finite")
    cosine = float(np.cos(np.deg2rad(angle)))
    sr, sp, positions = _seam_pairs(retained_faces, patch_faces, loop)
    normals, normal_valid = _unit_normals(vertices, retained_faces)
    triangles = vertices[retained_faces]
    double_area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                         triangles[:, 2] - triangles[:, 0]), axis=1)
    rn = normals[sr]
    vertex_faces = {}
    edge_faces = {}
    for fi, face in enumerate(retained_faces):
        for vertex in face:
            vertex_faces.setdefault(int(vertex), []).append(fi)
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            key = (min(int(a), int(b)), max(int(a), int(b)))
            edge_faces.setdefault(key, []).append(fi)
    vertex_edges = {}
    face_neighbors = [set() for _ in retained_faces]
    for edge, faces in edge_faces.items():
        for vertex in edge:
            vertex_edges.setdefault(vertex, []).append(edge)
        if len(faces) == 2:
            f, g = faces
            face_neighbors[f].add(g)
            face_neighbors[g].add(f)

    neighborhood_cache = {}
    sheet_cache = {}

    def vertex_neighborhood(fi):
        if fi not in neighborhood_cache:
            ids = set()
            for vertex in retained_faces[fi]:
                ids.update(vertex_faces[int(vertex)])
            neighborhood_cache[fi] = sorted(ids)
        return neighborhood_cache[fi]

    def observed_sheet(fi):
        """Locally evaluate the author's consistency metric on one normal sheet."""
        if fi in sheet_cache:
            return sheet_cache[fi]
        reference = normals[fi]
        best = None
        for center in vertex_neighborhood(fi):
            permitted = {j for j in vertex_neighborhood(center)
                         if normal_valid[j] and abs(float(normals[j] @ reference)) >= cosine}
            if fi not in permitted:
                continue
            connected = {fi}
            todo = [fi]
            while todo:
                current = todo.pop()
                for neighbor in face_neighbors[current]:
                    if neighbor in permitted and neighbor not in connected:
                        connected.add(neighbor)
                        todo.append(neighbor)
            ids = np.asarray(sorted(connected), dtype=np.int64)
            if len(ids) < 2:
                continue
            nn = normals[ids].copy()
            nn *= np.where(nn @ reference >= 0.0, 1.0, -1.0)[:, None]
            pair_distance = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * np.clip(nn @ nn.T, -1.0, 1.0)))
            maximum_range = float(pair_distance.max())
            lookup = {int(j): k for k, j in enumerate(ids)}
            variation = [float(pair_distance[lookup[j], lookup[k]])
                         for j in lookup for k in face_neighbors[j]
                         if k in lookup and j < k]
            if not variation:
                continue
            metric = maximum_range * max(variation) / (sum(variation) + 1e-9)
            mean = (double_area[ids, None] * nn).sum(axis=0)
            mean /= max(float(np.linalg.norm(mean)), 1e-15)
            reliable = (maximum_range <= sigma
                        and float(np.linalg.norm(mean - reference)) <= sigma)
            # Area and face index make equal-metric choices deterministic.
            ordering = (metric, -float(double_area[ids].sum()), int(center))
            if reliable and (best is None or ordering < best[0]):
                best = (ordering, mean, len(ids), maximum_range)
        result = (best[1], True, int(best[2]), float(best[3])) if best else (reference.copy(), False, 1, 0.0)
        sheet_cache[fi] = result
        return result

    counters = {
        "observed_sharp_edges_n": 0,
        "aligned_endpoint_edges_n": 0,
        "unreliable_endpoint_edges_n": 0,
        "side_mismatch_endpoint_edges_n": 0,
        "one_endpoint_supported_seam_edges_n": 0,
        "two_endpoints_supported_seam_edges_n": 0,
        "inconsistent_endpoint_seam_edges_n": 0,
        "ambiguous_seam_edges_n": 0,
    }
    sharp_observed = set()
    for edge, ids in edge_faces.items():
        if len(ids) == 2 and normal_valid[ids].all():
            if abs(float(normals[ids[0]] @ normals[ids[1]])) <= cosine:
                sharp_observed.add(edge)
    counters["observed_sharp_edges_n"] = len(sharp_observed)

    def endpoint_predictions(endpoint, outward, retained_normal):
        predictions = []
        for edge in vertex_edges.get(int(endpoint), []):
            if edge not in sharp_observed:
                continue
            other = edge[1] if edge[0] == endpoint else edge[0]
            direction = vertices[other] - vertices[endpoint]
            length = float(np.linalg.norm(direction))
            if length <= 1e-15 or float(direction @ outward) / length < cosine:
                continue
            counters["aligned_endpoint_edges_n"] += 1
            ids = edge_faces[edge]
            sheets = [observed_sheet(fi) for fi in ids]
            if not all(sheet[1] for sheet in sheets):
                counters["unreliable_endpoint_edges_n"] += 1
                continue
            nn = np.asarray([sheet[0] for sheet in sheets])
            nn *= np.where(nn @ retained_normal >= 0.0, 1.0, -1.0)[:, None]
            distances = np.linalg.norm(nn - retained_normal, axis=1)
            same = int(np.argmin(distances))
            alternate = nn[1 - same]
            if distances[same] > sigma or abs(float(alternate @ retained_normal)) > cosine:
                counters["side_mismatch_endpoint_edges_n"] += 1
                continue
            # A continuation normal must also be normal to the new seam edge.
            projected = alternate - float(alternate @ outward) * outward
            projected /= max(float(np.linalg.norm(projected)), 1e-15)
            if (float(np.linalg.norm(projected - alternate)) > sigma
                    or abs(float(projected @ retained_normal)) > cosine):
                counters["side_mismatch_endpoint_edges_n"] += 1
                continue
            predictions.append((projected, edge, float(distances[same])))
        return predictions

    sharp_mask = np.zeros(len(sr), dtype=bool)
    delta_target = np.zeros((len(sr), 3), dtype=np.float64)
    predicted_normal = rn.copy()
    support_edges = np.full((len(sr), 2, 2), -1, dtype=np.int64)
    for i, position in enumerate(positions):
        if not normal_valid[sr[i]]:
            continue
        a, b = int(loop[position]), int(loop[(position + 1) % len(loop)])
        tangent = vertices[b] - vertices[a]
        length = float(np.linalg.norm(tangent))
        if length <= 1e-15:
            continue
        tangent /= length
        left = endpoint_predictions(a, -tangent, rn[i])
        right = endpoint_predictions(b, tangent, rn[i])
        if bool(left) != bool(right):
            counters["one_endpoint_supported_seam_edges_n"] += 1
        if not left or not right:
            continue
        counters["two_endpoints_supported_seam_edges_n"] += 1
        candidates = []
        for nl, edge_l, error_l in left:
            for nr, edge_r, error_r in right:
                difference = float(np.linalg.norm(nl - nr))
                if difference > sigma:
                    continue
                target = nl + nr
                target /= max(float(np.linalg.norm(target)), 1e-15)
                if abs(float(target @ rn[i])) > cosine:
                    continue
                candidates.append((difference + error_l + error_r, edge_l, edge_r, target))
        if not candidates:
            counters["inconsistent_endpoint_seam_edges_n"] += 1
            continue
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        selected = candidates[0]
        if any(float(np.linalg.norm(item[3] - selected[3])) > sigma for item in candidates[1:]):
            counters["ambiguous_seam_edges_n"] += 1
            continue
        predicted_normal[i] = selected[3]
        delta_target[i] = selected[3] - rn[i]
        sharp_mask[i] = True
        support_edges[i, 0] = selected[1]
        support_edges[i, 1] = selected[2]
    variation_weight = np.ones(len(sr), dtype=np.float64)
    if len(sr) > 1:
        # Prevent a supported crease target from propagating to an unsupported edge.
        variation_weight[sharp_mask != np.roll(sharp_mask, -1)] = 0.0
        consecutive = (np.roll(positions, -1) - positions) % len(loop) == 1
        variation_weight[(sharp_mask | np.roll(sharp_mask, -1)) & ~consecutive] = 0.0
    elif len(sr):
        variation_weight[:] = 0.0
    retained_angle = (np.arccos(np.clip(np.abs(np.einsum("ij,ij->i", rn, np.roll(rn, -1, axis=0))), 0.0, 1.0))
                      if len(sr) else np.zeros(0, dtype=np.float64))
    return {
        "seam_retained": sr,
        "seam_patch": sp,
        "loop_edge_positions": positions,
        "orientation": np.ones(len(sr), dtype=np.float64),
        "normal_weight": np.ones(len(sr), dtype=np.float64),
        "delta_target": delta_target,
        "variation_weight": variation_weight,
        "sharp_mask": sharp_mask,
        "predicted_normal": predicted_normal,
        "support_edges": support_edges,
        "retained_transition_angle_radians": retained_angle,
        "summary": {
            "method": "observed_bilateral_crease_continuation",
            "feature_angle_degrees": angle,
            "continuation_angle_degrees": angle,
            "normal_range_sigma": sigma,
            "normal_guidance": "minimum_consistency_connected_observed_normal_sheet",
            "seam_edges_n": int(len(sr)),
            "sharp_edges_n": int(sharp_mask.sum()),
            "sharp_edge_fraction": float(sharp_mask.mean()) if len(sr) else 0.0,
            "variation_cuts_n": int((variation_weight == 0.0).sum()),
            "degenerate_retained_seam_edges_n": int((~normal_valid[sr]).sum()),
            "skipped_loop_edges_n": int(len(loop) - len(sr)),
            "observed_sheet_estimates_n": len(sheet_cache),
            "reliable_observed_sheet_estimates_n": sum(int(item[1]) for item in sheet_cache.values()),
            "frozen_from_initializer": False,
            "ground_truth_used": False,
            "patch_geometry_used_for_estimation": False,
            "sharp_target": "bilaterally_supported_observed_alternate_normal_jump",
            "smooth_target": "zero_normal_jump",
            **counters,
        },
    }
