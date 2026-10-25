"""Conservative IPC CCD filtering of fixed-topology vertex proposals.

This is a feasibility filter, not the IPC physical objective or solver.  The
caller keeps its optimizer, hard displacement projection, and final independent
changed-surface verification.  CCD is enabled only when the caller has an
explicit native-clear initializer and IPC finds no intersections involving a
movable primitive. Inherited intersections between fully fixed primitives are
excluded: their trajectories cannot change. This is not a claim that the full
input mesh is free of intersections.
Unsupported initial geometry is returned unchanged, with an explicit status.

The IPC Toolkit API and intersection-free initial-state requirement are from
https://github.com/ipc-sim/ipc-toolkit/blob/v1.4.0/python/src/ipc.cpp .
The native collision predicate and its primitive-wise OR expansion are from
python/src/collision_mesh.cpp and src/ipc/broad_phase/broad_phase.cpp in the
same official release. Every movable vertex receives a distinct positive
label; fixed vertices share label zero. Thus only fixed/fixed pairs are removed.
"""
from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path
import pickle
import queue
import struct
import subprocess
import sys
import threading
import time

import numpy as np


def _write_frame(stream, value):
    payload = pickle.dumps(value, protocol=4)
    stream.write(struct.pack("<Q", len(payload)))
    stream.write(payload)
    stream.flush()


def _read_exact(stream, length):
    chunks = []
    remaining = length
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError("Isolated IPC worker pipe closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_frame(stream):
    length = struct.unpack("<Q", _read_exact(stream, 8))[0]
    if length > 2 ** 32:
        raise ValueError("IPC worker returned an invalid frame length")
    return pickle.loads(_read_exact(stream, length))


class _IPCWorker:
    """Keep the IPC/TBB native runtime outside the PyTorch process on Windows."""

    def __init__(self):
        env = dict(os.environ, OMP_NUM_THREADS="1", TBB_NUM_THREADS="1")
        self.process = subprocess.Popen(
            [sys.executable, "-u", str(Path(__file__).resolve()), "--worker"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=env, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

    def exchange(self, request, timeout=60):
        _write_frame(self.process.stdin, request)
        mailbox = queue.Queue(maxsize=1)

        def read_response():
            try:
                mailbox.put((True, _read_frame(self.process.stdout)))
            except Exception as exc:
                mailbox.put((False, exc))

        threading.Thread(target=read_response, daemon=True).start()
        try:
            ok, response = mailbox.get(timeout=timeout)
        except queue.Empty:
            self.close()
            raise TimeoutError("Isolated IPC call exceeded 60 seconds")
        if not ok:
            self.process.wait(timeout=5)
            error = self.process.stderr.read().decode("utf-8", errors="replace")[-1000:]
            raise RuntimeError(f"Isolated IPC worker failed: {response}; {error}")
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "Unknown isolated IPC error"))
        return response

    def close(self):
        process = getattr(self, "process", None)
        if process is None:
            return
        if process.poll() is None:
            try:
                _write_frame(process.stdin, {"op": "quit"})
                process.wait(timeout=2)
            except Exception:
                process.kill()
                process.wait(timeout=5)
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                stream.close()
        self.process = None


def _worker_main():
    # The binary pipe is private, local, and only receives arrays serialized by
    # this module. No external or experiment-supplied pickle files are loaded.
    import ipctk
    mesh = None
    while True:
        try:
            request = _read_frame(sys.stdin.buffer)
        except EOFError:
            return
        if request["op"] == "quit":
            return
        try:
            if request["op"] == "initialize":
                vertices, faces = request["vertices"], request["faces"]
                mesh = ipctk.CollisionMesh(vertices, _unique_edges(faces), faces)
                labels = np.zeros(len(vertices), dtype=np.int32)
                variable = np.asarray(request["variable"], dtype=np.int64)
                labels[variable] = np.arange(1, len(variable) + 1, dtype=np.int32)
                mesh.can_collide = ipctk.VertexPatchesCanCollide(labels)
                result = {"ok": True, "has_intersections": bool(ipctk.has_intersections(mesh, vertices)),
                          "ipctk_version": importlib.metadata.version("ipctk")}
            elif request["op"] == "ccd":
                result = {"ok": True, "fraction": float(ipctk.compute_collision_free_stepsize(
                    mesh, request["current"], request["proposed"]))}
            else:
                raise ValueError("Unknown worker operation")
        except Exception as exc:
            result = {"ok": False, "error": repr(exc)}
        _write_frame(sys.stdout.buffer, result)


def _unique_edges(faces):
    return np.unique(np.sort(np.concatenate((faces[:, [0, 1]],
                                            faces[:, [1, 2]],
                                            faces[:, [2, 0]])), axis=1), axis=0)


def _crosses(vertices, faces):
    triangles = vertices[faces]
    return np.cross(triangles[:, 1] - triangles[:, 0],
                    triangles[:, 2] - triangles[:, 0])


def _first_projected_area_root(current, proposed, faces, reference_cross,
                               relative_floor):
    """First zero of projected doubled area along each linear trajectory.

    The normal reference is the initializer.  The projection is quadratic in
    step length, so taking the first positive real root also rejects a triangle
    that flips and flips back before the endpoint.  This is a conservative
    additional orientation constraint, separate from the CCD call.
    """
    if not len(faces):
        return 1.0
    triangles = current[faces]
    movement = (proposed - current)[faces]
    edge1 = triangles[:, 1] - triangles[:, 0]
    edge2 = triangles[:, 2] - triangles[:, 0]
    delta1 = movement[:, 1] - movement[:, 0]
    delta2 = movement[:, 2] - movement[:, 0]
    norm_squared = np.einsum("ij,ij->i", reference_cross, reference_cross)
    a = np.einsum("ij,ij->i", np.cross(delta1, delta2), reference_cross) / norm_squared
    b = np.einsum("ij,ij->i", np.cross(delta1, edge2) + np.cross(edge1, delta2), reference_cross) / norm_squared
    c = np.einsum("ij,ij->i", np.cross(edge1, edge2), reference_cross) / norm_squared - relative_floor
    if not np.isfinite(np.r_[a, b, c]).all() or np.any(c <= 0):
        return 0.0

    earliest = np.ones(len(faces), dtype=np.float64)
    # Scale-aware cancellation test; coefficients are normalized by rest area.
    linear = np.abs(a) <= 64 * np.finfo(float).eps * np.maximum(1.0, np.abs(b))
    valid_linear = linear & (b < 0)
    earliest[valid_linear] = np.minimum(1.0, -c[valid_linear] / b[valid_linear])
    discriminant = b * b - 4 * a * c
    quadratic = (~linear) & (discriminant >= 0)
    if np.any(quadratic):
        aa, bb, cc = a[quadratic], b[quadratic], c[quadratic]
        sqrt_disc = np.sqrt(discriminant[quadratic])
        # Numerically stable quadratic formula: q/a and c/q are the two roots.
        q = -.5 * (bb + np.copysign(sqrt_disc, bb))
        root1 = q / aa
        root2 = np.divide(cc, q, out=np.full_like(cc, np.inf), where=q != 0)
        root1[root1 <= 0] = np.inf
        root2[root2 <= 0] = np.inf
        earliest[quadratic] = np.minimum(1.0, np.minimum(root1, root2))
    return float(np.clip(earliest.min(initial=1.0), 0.0, 1.0))


class FeasibleStepGuard:
    """Limit each full-coordinate linear proposal using conservative CCD.

    Parameters
    ----------
    vertices, faces
        Initial full surface, including retained and patch triangles.
    variable_indices
        The exact vertices the caller may move.  The complementary vertices
        remain bitwise fixed. Only pairs of fully fixed primitives are excluded
        from IPC; every primitive involving a movable vertex stays eligible.
    initial_native_clear
        Must be the boolean True from successful independent initial geometry
        verification; absence, unknown, and false disable the guard.
    safety_fraction
        Strict fraction of a restricted step.  Full unrestricted steps retain
        length one.  The development protocol fixes this to 0.99.
    """

    def __init__(self, vertices, faces, variable_indices, *,
                 initial_native_clear=None, safety_fraction=.99,
                 relative_area_floor=1e-8):
        self.initial = np.array(vertices, dtype=np.float64, copy=True)
        self.faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3).copy()
        self.variable = np.unique(np.asarray(variable_indices, dtype=np.int64))
        if self.initial.ndim != 2 or self.initial.shape[1] != 3:
            raise ValueError("vertices must have shape (N, 3)")
        if not np.isfinite(self.initial).all() or not len(self.faces):
            raise ValueError("Initial vertices must be finite and surface nonempty")
        if self.faces.min() < 0 or self.faces.max() >= len(self.initial):
            raise ValueError("Face index is outside vertices")
        if len(self.variable) and (self.variable.min() < 0 or self.variable.max() >= len(self.initial)):
            raise ValueError("Variable index is outside vertices")
        if not 0 < safety_fraction < 1 or not 0 < relative_area_floor < 1:
            raise ValueError("Safety fraction and relative area floor must lie in (0,1)")
        self.safety_fraction = float(safety_fraction)
        self.relative_area_floor = float(relative_area_floor)
        self.fixed = np.setdiff1d(np.arange(len(self.initial)), self.variable)
        self.dynamic_faces = self.faces[np.any(np.isin(self.faces, self.variable), axis=1)]
        self.reference_cross = _crosses(self.initial, self.dynamic_faces)
        self.current = self.initial.copy()
        self.worker = None
        self._records = []
        self._info = {
            "status": "unsupported_initial_geometry", "enabled": False,
            "scope": "all IPC collision pairs involving at least one movable primitive; every linear accepted trajectory",
            "collision_pair_filter": "native VertexPatchesCanCollide: all fixed vertices label 0, each movable vertex a unique positive label",
            "excluded_collision_pairs": "fully fixed versus fully fixed primitives only",
            "initial_check_and_ccd_share_collision_filter": True,
            "global_absolute_collision_free_claim": False,
            "method": "IPC Toolkit conservative CCD plus quadratic projected-area constraint",
            "initial_native_clear": initial_native_clear is True,
            "initial_ipc_has_intersections": None,
            "initial_degenerate_faces_n": None,
            "safety_fraction": self.safety_fraction,
            "relative_projected_area_floor": self.relative_area_floor,
            "continuous_collision_free_claim": False,
            "final_independent_native_verification_required": True,
            "variable_vertices_n": int(len(self.variable)),
            "dynamic_faces_n": int(len(self.dynamic_faces)),
            "native_runtime": "one persistent isolated IPC worker per candidate; no PyTorch/native DLL mixing",
        }
        started = time.perf_counter()
        if initial_native_clear is not True:
            self._info["reason"] = "Initializer lacks explicit successful native-clear verification"
            self._info["setup_seconds"] = time.perf_counter() - started
            return
        all_cross = _crosses(self.initial, self.faces)
        degenerate = np.linalg.norm(all_cross, axis=1) <= 0
        self._info["initial_degenerate_faces_n"] = int(degenerate.sum())
        if degenerate.any():
            self._info["reason"] = "Initializer has degenerate triangles"
            self._info["setup_seconds"] = time.perf_counter() - started
            return
        try:
            self.worker = _IPCWorker()
            initialization = self.worker.exchange({"op": "initialize", "vertices": self.initial,
                                                  "faces": self.faces, "variable": self.variable})
            self._info["ipctk_version"] = initialization["ipctk_version"]
            has_intersections = initialization["has_intersections"]
            self._info["initial_ipc_has_intersections"] = has_intersections
            if has_intersections:
                self._info["reason"] = "IPC initial-intersection precondition fails for at least one movable primitive"
            else:
                self._info.update(status="ready", enabled=True,
                                  continuous_collision_free_claim=True,
                                  reason="Native clear, nondegenerate, and IPC intersection-free initializer over all pairs involving movable primitives")
        except (ImportError, OSError) as exc:
            self._info.update(status="dependency_unavailable", reason=repr(exc))
        except Exception as exc:
            self._info.update(status="initialization_failure", reason=repr(exc))
        self._info["setup_seconds"] = time.perf_counter() - started
        if not self._info["enabled"]:
            self.close()

    def filter(self, current_vertices, proposed_vertices):
        """Return an accepted full mesh and one-step diagnostic dictionary.

        Unsupported initialization returns the original mesh.  A runtime CCD
        error conservatively rejects the step.  Fixed coordinates are copied
        directly from the initializer, without floating-point reconstruction.
        The caller must project its hard displacement constraints before this
        call; interpolation then preserves those convex constraints.
        """
        current = np.asarray(current_vertices, dtype=np.float64)
        proposed = np.asarray(proposed_vertices, dtype=np.float64)
        if current.shape != self.initial.shape or proposed.shape != self.initial.shape:
            raise ValueError("Current and proposed coordinates must match the initial shape")
        if not np.array_equal(current, self.current):
            raise ValueError("Current coordinates differ from the last guard-accepted mesh")
        if not np.array_equal(proposed[self.fixed], self.initial[self.fixed]):
            raise ValueError("Proposed coordinates move vertices outside variable_indices")
        started = time.perf_counter()
        record = {"step": len(self._records) + 1, "status": "rejected",
                  "accepted_fraction": 0.0, "ccd_fraction": None,
                  "projected_area_fraction": None}
        accepted = current.copy()
        if not self._info["enabled"]:
            record.update(status="unsupported_initial_geometry", reason=self._info["reason"])
            accepted = self.initial.copy()
        elif not np.isfinite(proposed).all():
            record["reason"] = "Nonfinite proposal"
        elif np.array_equal(current, proposed):
            record.update(status="unchanged", accepted_fraction=1.0,
                          ccd_fraction=1.0, projected_area_fraction=1.0)
        else:
            try:
                area_fraction = _first_projected_area_root(
                    current, proposed, self.dynamic_faces, self.reference_cross,
                    self.relative_area_floor)
                ccd_fraction = self.worker.exchange({"op": "ccd", "current": current,
                                                     "proposed": proposed})["fraction"]
                if not np.isfinite(ccd_fraction) or not 0 <= ccd_fraction <= 1:
                    raise ValueError("CCD returned a nonfinite or out-of-range step")
                fraction = min(area_fraction, ccd_fraction)
                if fraction < 1:
                    fraction *= self.safety_fraction
                accepted[self.variable] = current[self.variable] + fraction * (proposed[self.variable] - current[self.variable])
                record.update(status="accepted" if fraction else "blocked",
                              accepted_fraction=float(fraction), ccd_fraction=ccd_fraction,
                              projected_area_fraction=area_fraction)
            except Exception as exc:
                record.update(status="ccd_failure", reason=repr(exc))
        accepted[self.fixed] = self.initial[self.fixed]
        self.current = accepted.copy()
        record["seconds"] = time.perf_counter() - started
        self._records.append(record)
        return accepted, dict(record)

    def diagnostics(self):
        """JSON-serializable setup, per-step fractions, and total filter cost."""
        return {**self._info, "steps": [dict(x) for x in self._records],
                "accepted_steps_n": sum(x["accepted_fraction"] > 0 for x in self._records),
                "restricted_steps_n": sum(x["accepted_fraction"] < 1 for x in self._records),
                "filter_seconds": sum(x["seconds"] for x in self._records)}

    def close(self):
        """Release the persistent native process after the final Adam proposal."""
        worker = getattr(self, "worker", None)
        if worker is not None:
            worker.close()
            self.worker = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        self.close()


if __name__ == "__main__":
    if sys.argv[1:] == ["--worker"]:
        _worker_main()
    else:
        raise SystemExit("This module only accepts its internal --worker invocation")
