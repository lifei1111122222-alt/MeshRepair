"""Observed feature chains, endpoint matching, and cross-hole cubic splines.

This is a deliberately bounded adaptation of Wang et al., *Automatic
Hole-Filling of CAD Models with Feature-Preserving*, Sections 4.1--4.3:
https://jjcao.weebly.com/uploads/4/5/3/4/4534726/accepte-online-automatic_hole-filling_of_cad_models_with_feature-preserving1.pdf

The ordered double normals, convexity test, inward-direction test, and
cumulative chord-length cubic interpolation follow that paper.  Differences
are explicit: a fixed 30 degree retained-edge graph replaces normal tensor
voting; a noncrossing dynamic program permits unmatched outliers instead of
assuming all endpoints fit the paper's nested pairing pattern. Ordered
normal discrepancies rank admissible matches without an added cutoff. Natural
end conditions complete the paper's unspecified cubic spline end conditions.
Only observed geometry is read.  Missing corners and feature junctions are
not inferred.  A separate triangulator and final native intersection check
remain necessary; noncrossing boundary indices do not prove 3D safety.
"""
from __future__ import annotations

import functools
import math

import numpy as np
from scipy.interpolate import CubicSpline


FEATURE_ANGLE_DEGREES = 30.0
MAX_CHAIN_EDGES = 6
MAX_ENDPOINTS = 128
MAX_CURVE_SEGMENTS = 512


def _edge_key(a, b):
    return (min(int(a), int(b)), max(int(a), int(b)))


def reconstruct_feature_curves(vertices, retained_faces, boundary_loop):
    """Return observed-only feature curves with exact original endpoints.

    Parameters are the damaged coordinates, retained triangles, and one
    ordered boundary loop.  No patch or reference geometry is accepted.

    The result contains ``curves`` and JSON-serializable ``diagnostics``.
    Each curve has ``endpoint_ids`` (two original vertex indices),
    ``endpoint_loop_positions``, ``points`` (N by 3, endpoints included),
    ``interior_points``, ``observed_chain_ids`` (two lists starting at their
    boundary endpoint), ``match_cost``, and JSON ``diagnostics``.  The caller
    assigns new global indices to each interior point once and uses the
    resulting shared polyline to split the boundary.  All returned endpoint
    pairs are noncrossing in cyclic boundary order and share no endpoints.

    Fixed choices: 30 degree unoriented feature threshold; paper ordered-normal
    matching cost; at most six observed chain edges; samples spaced at the
    damaged rim's median edge length.  Resource limits reject oversized
    cases rather than subsampling evidence or returning a truncated curve.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(retained_faces, dtype=np.int64).reshape(-1, 3)
    loop = np.asarray(boundary_loop, dtype=np.int64).reshape(-1)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must have shape (n, 3)")
    if len(loop) < 3 or len(set(map(int, loop))) != len(loop):
        raise ValueError("boundary_loop must contain at least three unique vertices")
    if (len(faces) == 0 or np.any(faces < 0) or np.any(faces >= len(vertices))
            or np.any(loop < 0) or np.any(loop >= len(vertices))):
        raise ValueError("invalid retained face or boundary vertex indices")
    observed_ids = np.unique(np.concatenate((faces.reshape(-1), loop)))
    if not np.isfinite(vertices[observed_ids]).all():
        raise ValueError("observed geometry must be finite")

    rim_lengths = np.linalg.norm(vertices[loop] - vertices[np.roll(loop, -1)], axis=1)
    if np.any(rim_lengths <= 0.0):
        raise ValueError("boundary_loop contains a zero-length edge")
    h = float(np.median(rim_lengths))
    eps_length = max(h * 1e-12, np.finfo(np.float64).tiny)
    angle_limit = math.radians(FEATURE_ANGLE_DEGREES)
    cosine_limit = math.cos(angle_limit)
    diag = {
        "method": "observed_endpoint_matching_and_chord_length_cubic_splines",
        "reference": "Wang et al., Automatic Hole-Filling of CAD Models with Feature-Preserving, Sections 4.1-4.3",
        "ground_truth_used": False,
        "patch_initializer_used": False,
        "feature_angle_degrees": FEATURE_ANGLE_DEGREES,
        "normal_match_angle_degrees": None,
        "normal_matching_rule": "paper ordered-double-normal cost, no additional angle cutoff",
        "max_chain_edges": MAX_CHAIN_EDGES,
        "sampling_median_rim_edge": h,
        "spline_boundary_conditions": "natural",
        "feature_extraction": "observed interior edge unoriented dihedral threshold",
        "matching": "maximum-cardinality noncrossing, then minimum ordered-double-normal distance",
        "max_endpoints": MAX_ENDPOINTS,
        "max_curve_segments": MAX_CURVE_SEGMENTS,
        "noncrossing_indices_only": True,
        "native_verification_required": True,
        "missing_corner_recovery": False,
        "status": "no_supported_curves",
    }

    tri = vertices[faces]
    crosses = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    areas2 = np.linalg.norm(crosses, axis=1)
    face_valid = areas2 > max(h * h * 1e-14, np.finfo(np.float64).tiny)
    normals = crosses / np.maximum(areas2[:, None], np.finfo(np.float64).tiny)
    edge_faces = {}
    neighbors = {}
    vertex_faces = {}
    for fi, face in enumerate(faces):
        for v in face:
            vertex_faces.setdefault(int(v), []).append(fi)
        for a, b in zip(face, np.roll(face, -1)):
            a, b = int(a), int(b)
            edge_faces.setdefault(_edge_key(a, b), []).append((fi, a, b))
            neighbors.setdefault(a, set()).add(b)
            neighbors.setdefault(b, set()).add(a)
    loop_edges = {_edge_key(a, b) for a, b in zip(loop, np.roll(loop, -1))}
    if any(len(edge_faces.get(edge, [])) != 1 for edge in loop_edges):
        raise ValueError("boundary_loop must follow retained one-face boundary edges")

    sharp_neighbors = {}
    sharp_edges = {}
    inconsistent_edges = 0
    for edge, records in edge_faces.items():
        if len(records) != 2:
            continue
        (f, a, b), (g, c, d) = records
        if not face_valid[f] or not face_valid[g]:
            continue
        if a != d or b != c:
            inconsistent_edges += 1
            continue
        dot = float(np.clip(normals[f] @ normals[g], -1.0, 1.0))
        if abs(dot) > cosine_limit:
            continue
        sharp_edges[edge] = records
        sharp_neighbors.setdefault(a, set()).add(b)
        sharp_neighbors.setdefault(b, set()).add(a)
    diag["retained_sharp_edges_n"] = len(sharp_edges)
    diag["inconsistently_oriented_edges_n"] = inconsistent_edges

    loop_set = set(map(int, loop))
    endpoints = []
    no_outward = 0
    ambiguous_outward = 0
    branch_stops = 0
    for position, raw_endpoint in enumerate(loop):
        endpoint = int(raw_endpoint)
        outward = sorted(v for v in sharp_neighbors.get(endpoint, ()) if v not in loop_set)
        if not outward:
            no_outward += 1
            continue
        if len(outward) != 1:
            ambiguous_outward += 1
            continue
        chain = [endpoint, outward[0]]
        stop_reason = "maximum_observed_chain_length"
        while len(chain) - 1 < MAX_CHAIN_EDGES:
            current = chain[-1]
            options = sorted(v for v in sharp_neighbors.get(current, ()) if v != chain[-2])
            if len(options) != 1:
                stop_reason = "branch" if options else "feature_end"
                branch_stops += int(bool(options))
                break
            nxt = options[0]
            if nxt in loop_set or nxt in chain:
                stop_reason = "boundary_or_closed_chain"
                break
            chain.append(nxt)
        first = chain[1]
        records = sharp_edges[_edge_key(endpoint, first)]
        forward = next((fi for fi, a, b in records if a == endpoint and b == first), None)
        backward = next((fi for fi, a, b in records if a == first and b == endpoint), None)
        if forward is None or backward is None:
            continue
        normal_pair = np.stack((normals[forward], normals[backward]))
        incident_ids = np.asarray(vertex_faces[endpoint], dtype=np.int64)
        normal_sum = normals[incident_ids[face_valid[incident_ids]]].sum(axis=0)
        normal_length = float(np.linalg.norm(normal_sum))
        if normal_length <= 1e-12:
            continue
        mean_normal = normal_sum / normal_length
        neighbor_ids = np.asarray(sorted(neighbors[endpoint]), dtype=np.int64)
        laplacian = vertices[neighbor_ids].mean(axis=0) - vertices[endpoint]
        convexity_projection = float(mean_normal @ laplacian)
        endpoints.append({
            "id": endpoint, "position": int(position), "chain": chain,
            "normal_pair": normal_pair,
            "convex": bool(convexity_projection <= 0.0),
            "convexity_projection": convexity_projection,
            "inward": vertices[endpoint] - vertices[first],
            "chain_stop": stop_reason,
        })
    diag.update({
        "boundary_vertices_n": len(loop),
        "boundary_no_outward_feature_n": no_outward,
        "boundary_ambiguous_outward_feature_n": ambiguous_outward,
        "observed_chain_branch_stops_n": branch_stops,
        "feature_endpoints_n": len(endpoints),
        "feature_endpoints": [{"vertex_id": p["id"], "loop_position": p["position"],
            "observed_chain_ids": p["chain"], "chain_stop": p["chain_stop"],
            "convex": p["convex"], "convexity_projection": p["convexity_projection"]}
            for p in endpoints],
    })
    if len(endpoints) > MAX_ENDPOINTS:
        diag["status"] = "endpoint_resource_limit"
        return {"curves": [], "diagnostics": diag}

    feasible = {}
    rejection_counts = {"adjacent_boundary": 0, "convexity": 0, "inward_direction": 0,
                        "ordered_normals": 0, "same_observed_chain": 0}
    for i, left in enumerate(endpoints):
        for j in range(i + 1, len(endpoints)):
            right = endpoints[j]
            if (right["position"] - left["position"]) % len(loop) in (1, len(loop) - 1):
                rejection_counts["adjacent_boundary"] += 1
                continue
            if left["convex"] != right["convex"]:
                rejection_counts["convexity"] += 1
                continue
            if set(left["chain"]).intersection(right["chain"]):
                rejection_counts["same_observed_chain"] += 1
                continue
            chord = vertices[right["id"]] - vertices[left["id"]]
            length = float(np.linalg.norm(chord))
            if (length <= eps_length or float(chord @ left["inward"]) <= 0.0
                    or float((-chord) @ right["inward"]) <= 0.0):
                rejection_counts["inward_direction"] += 1
                continue
            ordered_dot = np.einsum("ij,ij->i", left["normal_pair"], right["normal_pair"][::-1])
            angles = np.arccos(np.clip(ordered_dot, -1.0, 1.0))
            cost = float(np.linalg.norm(left["normal_pair"] - right["normal_pair"][::-1], axis=1).sum())
            feasible[i, j] = {"cost": cost, "normal_angles": angles,
                              "chord_length": length}
    diag["pair_rejections"] = rejection_counts
    diag["compatible_endpoint_pairs_n"] = len(feasible)

    @functools.lru_cache(maxsize=None)
    def solve(first, last):
        if first >= last:
            return (0, 0.0, ())
        # Lexicographic objective: most supported pairs, then smallest normal
        # discrepancy, then stable pair indices.  Unmatched endpoints remain
        # available to the ordinary whole-hole fallback.
        best = solve(first + 1, last)
        for other in range(first + 1, last + 1):
            match = feasible.get((first, other))
            if match is None:
                continue
            inner = solve(first + 1, other - 1)
            outer = solve(other + 1, last)
            candidate = (inner[0] + outer[0] - 1,
                         inner[1] + outer[1] + match["cost"],
                         ((first, other),) + inner[2] + outer[2])
            if candidate < best:
                best = candidate
        return best

    pairs = tuple(sorted(solve(0, len(endpoints) - 1)[2])) if endpoints else ()
    curves = []
    spline_rejections = {"nonfinite_or_degenerate": 0, "resource_limit": 0,
                         "outward_spline_tangent": 0}
    for i, j in pairs:
        left, right = endpoints[i], endpoints[j]
        chain_ids = left["chain"][::-1] + right["chain"]
        knots = vertices[np.asarray(chain_ids, dtype=np.int64)]
        spacings = np.linalg.norm(np.diff(knots, axis=0), axis=1)
        if np.any(spacings <= eps_length):
            spline_rejections["nonfinite_or_degenerate"] += 1
            continue
        parameters = np.concatenate(([0.0], np.cumsum(spacings)))
        spline = CubicSpline(parameters, knots, axis=0, bc_type="natural")
        start = float(parameters[len(left["chain"]) - 1])
        end = float(parameters[len(left["chain"])])
        chord = feasible[i, j]["chord_length"]
        coarse_segments = max(2, int(math.ceil(chord / h)))
        if coarse_segments > MAX_CURVE_SEGMENTS:
            spline_rejections["resource_limit"] += 1
            continue
        dense_t = np.linspace(start, end, 16 * coarse_segments + 1)
        dense_points = spline(dense_t)
        if not np.isfinite(dense_points).all():
            spline_rejections["nonfinite_or_degenerate"] += 1
            continue
        arc_lengths = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(dense_points, axis=0), axis=1))))
        curve_length = float(arc_lengths[-1])
        segments = max(2, int(math.ceil(curve_length / h)))
        if segments > MAX_CURVE_SEGMENTS:
            spline_rejections["resource_limit"] += 1
            continue
        sample_t = np.interp(np.linspace(0.0, curve_length, segments + 1), arc_lengths, dense_t)
        points = np.asarray(spline(sample_t), dtype=np.float64)
        points[0] = vertices[left["id"]]
        points[-1] = vertices[right["id"]]
        if (not np.isfinite(points).all()
                or np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) <= eps_length)):
            spline_rejections["nonfinite_or_degenerate"] += 1
            continue
        if (float((points[1] - points[0]) @ left["inward"]) <= 0.0
                or float((points[-2] - points[-1]) @ right["inward"]) <= 0.0):
            spline_rejections["outward_spline_tangent"] += 1
            continue
        curves.append({
            "endpoint_ids": (left["id"], right["id"]),
            "endpoint_loop_positions": (left["position"], right["position"]),
            "points": points,
            "interior_points": points[1:-1],
            "observed_chain_ids": (list(left["chain"]), list(right["chain"])),
            "match_cost": feasible[i, j]["cost"],
            "diagnostics": {
                "ordered_normal_angle_radians": feasible[i, j]["normal_angles"].tolist(),
                "convex": left["convex"],
                "chord_length": chord,
                "curve_length": curve_length,
                "segments_n": segments,
                "observed_knot_count": len(knots),
                "interpolation_max_knot_error": float(np.max(np.abs(spline(parameters) - knots))),
            },
        })
    diag.update({
        "matched_endpoint_pairs_n": len(pairs),
        "selected_endpoint_pairs": [[endpoints[i]["id"], endpoints[j]["id"]] for i, j in pairs],
        "unmatched_feature_endpoints_n": len(endpoints) - 2 * len(pairs),
        "spline_rejections": spline_rejections,
        "curves_n": len(curves),
        "curve_interior_vertices_n": sum(len(c["interior_points"]) for c in curves),
        "status": "curves_reconstructed" if curves else "no_supported_curves",
    })
    return {"curves": curves, "diagnostics": diag}
