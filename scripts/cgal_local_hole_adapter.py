"""Target-only adapter to the official CGAL 6.0.1 Python bindings.

Only damaged vertices, damaged faces and a chosen boundary are accepted. No
annotation, removed geometry, selector features, or score enters the adapter.
The native package lives in an isolated directory, not the main environment.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CGAL_DIR = ROOT / ".external_baselines" / "cgal601"
METHODS = ("cgal_polyline", "cgal_refine_fair", "feature_partition")


class UnsupportedFeaturePartition(ValueError):
    """The observed neighborhood does not support a feature partition."""


def _imports():
    sys.path.insert(0, str(CGAL_DIR))
    from CGAL.CGAL_Kernel import Point_3
    from CGAL.CGAL_Polyhedron_3 import Polyhedron_3, Polyhedron_modifier
    from CGAL import CGAL_Polygon_mesh_processing as pmp
    return Point_3, Polyhedron_3, Polyhedron_modifier, pmp


def metadata():
    _, _, _, pmp = _imports()
    return {
        "library": "CGAL Polygon Mesh Processing",
        "binding_distribution": "cgal",
        "binding_version": importlib.metadata.version("cgal"),
        "cgal_version_number": pmp.CGAL_VERSION_NR,
        "cgal_release_date": pmp.CGAL_RELEASE_DATE,
        "license": "GPL-3.0-or-later (wheel metadata); commercial license also offered",
        "repository": "https://github.com/CGAL/cgal-swig-bindings",
        "official_docs": "https://doc.cgal.org/6.0.1/Polygon_mesh_processing/group__PMP__hole__filling__grp.html",
        "method_api": {
            "cgal_polyline": "CGAL::Polygon_mesh_processing::triangulate_hole_polyline(points, third_points, out)",
            "cgal_refine_fair": "CGAL::Polygon_mesh_processing::triangulate_refine_and_fair_hole(mesh, target_halfedge, faces_out, vertices_out)",
            "feature_partition": "CGAL::Polygon_mesh_processing::triangulate_hole_polyline(points, out), refine(mesh, facets, faces_out, vertices_out), fair(mesh, interior_vertices)",
        },
        "parameters": "Official defaults: density_control_factor=sqrt(2), fairing_continuity=1; triangulation parameters not exposed by binding",
        "fairing_status_observable": False,
        "fairing_status_reason": "SWIG wrapper returns void and discards CGAL's success boolean; successful output does not certify fairing solver success",
    }


def _validate_boundary(vertices, faces, loop):
    if len(loop) < 3 or len(np.unique(loop)) != len(loop):
        raise ValueError("selected boundary must be a simple cycle with at least 3 distinct vertices")
    if not np.isfinite(vertices).all() or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("nonfinite vertices or invalid face indices")
    if loop.min() < 0 or loop.max() >= len(vertices):
        raise ValueError("invalid selected boundary indices")
    target_edges = {tuple(sorted((int(a), int(b)))) for a, b in zip(loop, np.roll(loop, -1))}
    incident = defaultdict(list)
    for fi, face in enumerate(faces):
        for a, b, c in ((face[0], face[1], face[2]),
                        (face[1], face[2], face[0]),
                        (face[2], face[0], face[1])):
            key = tuple(sorted((int(a), int(b))))
            if key in target_edges:
                incident[key].append((fi, int(a), int(b), int(c)))
    if any(len(incident[e]) != 1 for e in target_edges):
        raise ValueError("selected loop is not exactly a cycle of mesh boundary edges")
    return incident


def _target_component(faces, boundary_incident):
    """Retain the whole edge-connected component(s) touching the selected seam.

    Unrelated disconnected components cannot influence CGAL fairing, and need
    not satisfy Polyhedron_3's manifold precondition. Nothing is deleted from
    the returned repaired mesh: this component is solely the native context.
    """
    edge_faces = defaultdict(list)
    for fi, face in enumerate(faces):
        for a, b in zip(face, np.roll(face, -1)):
            edge_faces[tuple(sorted((int(a), int(b))))].append(fi)
    seeds = {entry[0][0] for entry in boundary_incident.values()}
    seen = set(seeds)
    q = deque(sorted(seeds))
    while q:
        fi = q.popleft()
        for a, b in zip(faces[fi], np.roll(faces[fi], -1)):
            for other in edge_faces[tuple(sorted((int(a), int(b))))]:
                if other not in seen:
                    seen.add(other)
                    q.append(other)
    return faces[np.asarray(sorted(seen), dtype=np.int64)]


def _polyhedron(vertices, faces, Polyhedron_3, Polyhedron_modifier, Point_3):
    # Check edge manifoldness and winding before calling a native builder;
    # vertex-manifold defects are still reported by its validity check.
    edges = defaultdict(list)
    for face in faces:
        if len(set(face.tolist())) != 3:
            raise ValueError("native context contains a repeated-vertex face")
        for a, b in zip(face, np.roll(face, -1)):
            edges[tuple(sorted((int(a), int(b))))].append((int(a), int(b)))
    if any(len(es) > 2 for es in edges.values()):
        raise ValueError("native context is nonmanifold at an edge")
    if any(len(es) == 2 and es[0] == es[1] for es in edges.values()):
        raise ValueError("native context has inconsistent face winding")
    mod = Polyhedron_modifier()
    mod.begin_surface(len(vertices), len(faces))
    for point in vertices:
        mod.add_vertex(Point_3(*map(float, point)))
    for face in faces:
        mod.begin_facet()
        for vi in face:
            mod.add_vertex_to_facet(int(vi))
        mod.end_facet()
    mod.end_surface()
    poly = Polyhedron_3()
    poly.delegate(mod)
    if poly.size_of_vertices() != len(vertices) or poly.size_of_facets() != len(faces) or not poly.is_valid():
        raise ValueError("CGAL rejected input topology; input cleaning is prohibited")
    for i, v in enumerate(poly.vertices()):
        v.set_id(i)
    return poly


def _assert_target_only(vertices, original_faces, loop, patch):
    from collections import Counter
    target = {tuple(sorted((int(a), int(b)))) for a, b in zip(loop, np.roll(loop, -1))}
    counts = Counter(tuple(sorted((int(a), int(b)))) for f in patch for a, b in zip(f, np.roll(f, -1)))
    if not len(patch) or {e for e, n in counts.items() if n == 1} != target or any(n > 2 for n in counts.values()):
        raise ValueError("CGAL patch boundary does not equal selected loop")
    retained_counts = Counter(tuple(sorted((int(a), int(b)))) for f in original_faces for a, b in zip(f, np.roll(f, -1)))
    if any(e not in target and retained_counts[e] > 0 for e in counts):
        raise ValueError("patch unexpectedly shares retained edges beyond the selected seam")
    if patch.min() < 0 or patch.max() >= len(vertices) or not np.isfinite(vertices).all():
        raise ValueError("invalid patch geometry")


def _assert_patch_winding(original_faces, loop, patch, feature_edges=()):
    """Check every shared edge, including the introduced subdivision curves."""
    incident = defaultdict(list)
    for face in patch:
        for a, b in zip(face, np.roll(face, -1)):
            incident[tuple(sorted((int(a), int(b))))].append((int(a), int(b)))
    for edge, oriented in incident.items():
        if len(oriented) == 2 and oriented[0] != oriented[1][::-1]:
            raise ValueError("piecewise patch has inconsistent internal winding")
        if len(oriented) > 2:
            raise ValueError("piecewise patch has a nonmanifold edge")
    target = {tuple(sorted((int(a), int(b)))) for a, b in zip(loop, np.roll(loop, -1))}
    for face in original_faces:
        for a, b in zip(face, np.roll(face, -1)):
            key = tuple(sorted((int(a), int(b))))
            if key in target and incident[key] != [(int(b), int(a))]:
                raise ValueError("piecewise patch winding disagrees with retained seam")
    if any(len(incident[tuple(sorted(map(int, edge)))]) != 2 for edge in feature_edges):
        raise ValueError("a feature curve is not an internal two-sided patch edge")


def _split_feature_subloops(loop, curve_paths):
    """Split an oriented polygon along noncrossing paths without duplicating IDs."""
    subloops = [list(map(int, loop))]
    for path in curve_paths:
        a, b = int(path[0]), int(path[-1])
        containing = [i for i, sub in enumerate(subloops) if a in sub and b in sub]
        if len(containing) != 1:
            raise ValueError("feature endpoints do not define one noncrossing subhole")
        sub = subloops.pop(containing[0])
        ia, ib = sub.index(a), sub.index(b)
        rotated = sub[ia:] + sub[:ia]
        ib = rotated.index(b)
        left = rotated[:ib + 1] + list(reversed(path[1:-1]))
        right = rotated[ib:] + rotated[:1] + list(path[1:-1])
        if any(len(s) < 3 or len(set(s)) != len(s) for s in (left, right)):
            raise ValueError("feature partition creates a degenerate subloop")
        subloops.extend((left, right))
    return subloops


def repair_feature_partition(vertices, faces, selected_loop):
    """Reconstruct observable feature curves and fill their subholes with CGAL.

    Curve boundaries and retained input coordinates are immutable. Each subhole
    uses the official polyline triangulator, default refinement, and fairing of
    newly inserted interior vertices only. The no-third-point overload is used:
    inventing an adjacent face across an unfilled feature edge would introduce
    an unsupported normal. This is a CGAL adaptation of feature subdivision,
    not a reproduction of the MAFM algorithm in Wang et al. (2012).
    """
    sys.path.insert(0, str(ROOT))
    from tcmr.feature_curves import reconstruct_feature_curves

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    loop = np.asarray(selected_loop, dtype=np.int64)
    incident = _validate_boundary(vertices, faces, loop)
    proposal = reconstruct_feature_curves(vertices, faces, loop)
    curves = proposal["curves"]
    if not curves:
        exc = UnsupportedFeaturePartition("no supported cross-hole feature pair")
        exc.diagnostics = proposal["diagnostics"]
        raise exc
    # A patch traverses each rim edge opposite to its retained incident face.
    first = incident[tuple(sorted((int(loop[0]), int(loop[1]))))][0]
    if (int(loop[0]), int(loop[1])) == (first[1], first[2]):
        oriented_loop = loop[::-1]
    else:
        oriented_loop = loop
    current_vertices = vertices.copy()
    curve_paths = []
    curve_info = []
    for curve in curves:
        a, b = map(int, curve["endpoint_ids"])
        points = np.asarray(curve["points"], dtype=np.float64)
        if (points.ndim != 2 or points.shape[1] != 3 or len(points) < 2
                or not np.isfinite(points).all()
                or not np.array_equal(points[0], vertices[a])
                or not np.array_equal(points[-1], vertices[b])):
            raise ValueError("feature reconstruction changed an original endpoint")
        if np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) <= 1e-14):
            raise ValueError("feature curve has a zero-length edge")
        inner_ids = list(range(len(current_vertices), len(current_vertices) + len(points) - 2))
        current_vertices = np.concatenate((current_vertices, points[1:-1]), axis=0)
        path = [a] + inner_ids + [b]
        curve_paths.append(path)
        curve_info.append({
            "vertex_ids": path,
            "endpoint_ids": [a, b],
            "match_cost": float(curve["match_cost"]),
            "observed_chain_ids": curve["observed_chain_ids"],
            "diagnostics": curve["diagnostics"],
        })
    subloops = _split_feature_subloops(oriented_loop, curve_paths)
    feature_ids = sorted(set(i for path in curve_paths for i in path))
    feature_points = current_vertices[feature_ids].copy()
    feature_edges = [[int(a), int(b)] for path in curve_paths for a, b in zip(path[:-1], path[1:])]
    Point_3, Polyhedron_3, Polyhedron_modifier, pmp = _imports()
    patches, subhole_info = [], []
    for subloop in subloops:
        ids = np.asarray(subloop, dtype=np.int64)
        points = [Point_3(*map(float, p)) for p in current_vertices[ids]]
        triples = []
        pmp.triangulate_hole_polyline(points, triples)
        local_patch = np.asarray([[t.first, t.second, t.third] for t in triples], dtype=np.int64).reshape(-1, 3)
        if len(local_patch) != len(ids) - 2:
            raise ValueError("CGAL did not triangulate the full feature subhole")
        # The native triangulator's orientation is tied to the point ordering.
        edges = [(int(a), int(b)) for f in local_patch for a, b in zip(f, np.roll(f, -1))]
        if (1, 0) in edges:
            local_patch = local_patch[:, [0, 2, 1]]
        if (0, 1) not in [(int(a), int(b)) for f in local_patch for a, b in zip(f, np.roll(f, -1))]:
            raise ValueError("CGAL subhole lacks its first boundary edge")
        boundary_points = current_vertices[ids].copy()
        poly = _polyhedron(boundary_points, local_patch, Polyhedron_3, Polyhedron_modifier, Point_3)
        new_faces, new_vertices = [], []
        pmp.refine(poly, list(poly.facets()), new_faces, new_vertices)
        fair_result = None
        if new_vertices:
            fair_result = pmp.fair(poly, new_vertices)
            if fair_result is False:
                raise ValueError("CGAL interior fairing failed")
        handles = list(poly.vertices())
        after_boundary = np.asarray([[h.point().x(), h.point().y(), h.point().z()] for h in handles[:len(ids)]])
        if not np.array_equal(after_boundary, boundary_points):
            raise ValueError("CGAL moved a fixed feature or rim vertex")
        for handle, global_id in zip(handles[:len(ids)], ids):
            handle.set_id(int(global_id))
        added = []
        for i, handle in enumerate(handles[len(ids):]):
            handle.set_id(len(current_vertices) + i)
            p = handle.point()
            added.append([p.x(), p.y(), p.z()])
        current_vertices = np.concatenate((current_vertices, np.asarray(added, dtype=np.float64).reshape(-1, 3)), axis=0)
        patch = []
        for facet in poly.facets():
            h = facet.halfedge()
            patch.append([h.vertex().id(), h.next().vertex().id(), h.next().next().vertex().id()])
        patch = np.asarray(patch, dtype=np.int64).reshape(-1, 3)
        _assert_target_only(current_vertices, np.empty((0, 3), dtype=np.int64), ids, patch)
        patches.append(patch)
        subhole_info.append({"boundary_vertices": len(ids), "patch_faces": len(patch),
                             "added_interior_vertices": len(added),
                             "fairing_return": None if fair_result is None else bool(fair_result)})
    patch = np.concatenate(patches, axis=0)
    _assert_target_only(current_vertices, faces, loop, patch)
    _assert_patch_winding(faces, loop, patch, feature_edges)
    if not np.array_equal(current_vertices[:len(vertices)], vertices):
        raise ValueError("feature partition changed retained input coordinates")
    if not np.array_equal(current_vertices[feature_ids], feature_points):
        raise ValueError("feature partition changed a reconstructed curve")
    details = {
        "method": "feature_partition", "target_only": True,
        "retained_vertices_unchanged": True, "feature_vertices_unchanged": True,
        "fallback_used": False, "curve_count": len(curves), "subloop_count": len(subloops),
        "feature_vertex_ids": feature_ids, "feature_edges": feature_edges,
        "curves": curve_info, "curve_reconstruction": proposal["diagnostics"],
        "subholes": subhole_info, "added_vertices": len(current_vertices) - len(vertices),
        "third_points_used": False,
        "filling": "CGAL default constrained polyline triangulation, refinement, interior-only fairing",
        "method_scope": "feature-subdivision adaptation; not an implementation of Wang MAFM",
    }
    return current_vertices, patch, details


def repair(vertices, faces, selected_loop, method="cgal_refine_fair"):
    if method == "feature_partition":
        return repair_feature_partition(vertices, faces, selected_loop)
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    loop = np.asarray(selected_loop, dtype=np.int64)
    incident = _validate_boundary(vertices, faces, loop)
    Point_3, Polyhedron_3, Polyhedron_modifier, pmp = _imports()
    details = {"method": method, "target_only": True, "retained_vertices_unchanged": True,
               "fallback_used": False, "fairing_success": None}
    if method == "cgal_polyline":
        points = [Point_3(*map(float, p)) for p in vertices[loop]]
        thirds = [Point_3(*map(float, vertices[incident[tuple(sorted((int(a), int(b))))][0][3]]))
                  for a, b in zip(loop, np.roll(loop, -1))]
        triples = []
        pmp.triangulate_hole_polyline(points, thirds, triples)
        patch = np.asarray([[loop[t.first], loop[t.second], loop[t.third]] for t in triples], dtype=np.int64).reshape(-1, 3)
        out_vertices = vertices.copy()
        details["third_points_from_retained_faces"] = True
    elif method == "cgal_refine_fair":
        component = _target_component(faces, incident)
        ids = np.unique(component)
        local_ids = np.full(len(vertices), -1, dtype=np.int64)
        local_ids[ids] = np.arange(len(ids))
        poly = _polyhedron(vertices[ids], local_ids[component], Polyhedron_3, Polyhedron_modifier, Point_3)
        target_edges = {tuple(sorted((int(local_ids[a]), int(local_ids[b])))) for a, b in zip(loop, np.roll(loop, -1))}
        starts = [h for h in poly.halfedges() if h.is_border()
                  and tuple(sorted((h.vertex().id(), h.opposite().vertex().id()))) in target_edges]
        if len(starts) != len(loop):
            raise ValueError("CGAL boundary differs from selected loop")
        start = starts[0]
        walked = set()
        h = start
        while True:
            walked.add(tuple(sorted((h.vertex().id(), h.opposite().vertex().id()))))
            h = h.next()
            if h == start:
                break
            if len(walked) > len(loop):
                raise ValueError("CGAL border includes another boundary")
        if walked != target_edges:
            raise ValueError("CGAL border cycle does not match selected loop")
        native_faces, native_vertices = [], []
        pmp.triangulate_refine_and_fair_hole(poly, start, native_faces, native_vertices)
        original_handles = list(poly.vertices())[:len(ids)]
        original_points = np.asarray([[v.point().x(), v.point().y(), v.point().z()] for v in original_handles])
        if not np.array_equal(original_points, vertices[ids]):
            raise ValueError("CGAL modified retained vertices")
        # CGAL appends new vertices. Identify by stable original handle ids,
        # then explicitly assign all handles their global output ids.
        for handle, original_id in zip(original_handles, ids):
            handle.set_id(int(original_id))
        added_points = []
        for i, v in enumerate(list(poly.vertices())[len(ids):]):
            v.set_id(len(vertices) + i)
            p = v.point()
            added_points.append([p.x(), p.y(), p.z()])
        out_vertices = np.concatenate([vertices, np.asarray(added_points).reshape(-1, 3)], axis=0)
        patch = []
        for f in native_faces:
            h = f.halfedge()
            patch.append([h.vertex().id(), h.next().vertex().id(), h.next().next().vertex().id()])
        patch = np.asarray(patch, dtype=np.int64).reshape(-1, 3)
        details.update(native_context_faces=len(component), native_context_vertices=len(ids),
                       added_vertices=len(added_points), fairing_status_unobservable_in_binding=True)
    else:
        raise ValueError(method)
    _assert_target_only(out_vertices, faces, loop, patch)
    return out_vertices, patch, details


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="NPZ with vertices, faces, selected_loop only")
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    args = parser.parse_args()
    output = Path(args.output)
    try:
        with np.load(args.input, allow_pickle=False) as data:
            vv, pf, details = repair(data["vertices"], data["faces"], data["selected_loop"], args.method)
        np.savez_compressed(output.with_suffix(".npz"), vertices=vv, patch_faces=pf)
        result = {"status": "success", "details": details, "package": metadata()}
    except UnsupportedFeaturePartition as exc:
        result = {"status": "unsupported", "error": str(exc),
                  "details": getattr(exc, "diagnostics", {})}
    except Exception as exc:
        result = {"status": "adapter_failure", "error": str(exc), "error_type": type(exc).__name__}
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
