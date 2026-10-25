"""Strict input-cycle and local patch topology diagnostics.

These graph checks do not certify geometric nonintersection. They are kept
separate from legacy closure counts and never overwrite historical results.
"""
from collections import Counter, defaultdict

import numpy as np


def incidence(faces):
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    raw = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    unique, inverse, counts = np.unique(np.sort(raw, axis=1), axis=0,
                                         return_inverse=True, return_counts=True)
    orientation = np.bincount(inverse, weights=np.where(raw[:,0]<raw[:,1], 1, -1), minlength=len(unique))
    return {tuple(map(int, e)): (int(n), int(o)) for e, n, o in zip(unique, counts, orientation)}


def cycle_edges(loop):
    loop = np.asarray(loop, dtype=np.int64).reshape(-1)
    return [tuple(sorted((int(a), int(b)))) for a, b in zip(loop, np.roll(loop, -1))]


def inspect_target_cycle(faces, loop):
    loop = np.asarray(loop, dtype=np.int64).reshape(-1)
    table = incidence(faces)
    edges = cycle_edges(loop)
    boundary_degree = Counter(v for e, (n, _) in table.items() if n == 1 for v in e)
    unique_vertices = len(set(map(int, loop))) == len(loop)
    edge_counts = [table.get(e, (0, 0))[0] for e in edges]
    degrees = [boundary_degree[int(v)] for v in loop]
    valid = (len(loop)>=3 and unique_vertices and all(n==1 for n in edge_counts)
             and all(d==2 for d in degrees))
    return {'input_simple_closed_boundary': bool(valid),
            'loop_vertices': int(len(loop)), 'unique_loop_vertices': bool(unique_vertices),
            'cyclic_edges_not_input_boundary': int(sum(n!=1 for n in edge_counts)),
            'loop_vertices_with_boundary_degree_not_two': int(sum(d!=2 for d in degrees))}


def _patch_vertex_links(patch, boundary_vertices):
    links = defaultdict(list)
    for a, b, c in patch:
        links[int(a)].append((int(b), int(c)))
        links[int(b)].append((int(c), int(a)))
        links[int(c)].append((int(a), int(b)))
    bad = 0
    for vertex, edges in links.items():
        adj = defaultdict(list)
        for a, b in edges:
            adj[a].append(b); adj[b].append(a)
        visited, stack = set(), [next(iter(adj))]
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node); stack.extend(adj[node])
        deg = [len(v) for v in adj.values()]
        degree_ok = (deg.count(1)==2 and all(d in (1, 2) for d in deg)
                     if vertex in boundary_vertices else all(d==2 for d in deg))
        if not degree_ok or len(visited)!=len(adj):
            bad += 1
    return bad


def inspect_local_patch(retained_faces, loop, patch_faces):
    """Return closure and disk/winding diagnostics using only mesh topology.

    Retained faces use their input vertex IDs. Output retained connectivity
    must be unchanged; coordinate deformation is measured separately.
    """
    retained = np.asarray(retained_faces, dtype=np.int64).reshape(-1, 3)
    patch = np.asarray(patch_faces, dtype=np.int64).reshape(-1, 3)
    result = inspect_target_cycle(retained, loop)
    target = set(cycle_edges(loop))
    ptable = incidence(patch)
    rtable = incidence(retained)
    boundary = {e for e, (n, _) in ptable.items() if n==1}
    bad_count = sum(n not in (1, 2) for n, _ in ptable.values())
    repeated_faces = len(patch)-len({tuple(sorted(map(int, f))) for f in patch})
    repeated_vertices = sum(len(set(map(int, f)))!=3 for f in patch)
    exact_boundary = bool(len(patch) and boundary==target)
    seam_two = all(rtable.get(e,(0,0))[0]+ptable.get(e,(0,0))[0]==2 for e in target)
    opposing_seam = exact_boundary and all(rtable.get(e,(0,0))[1]+ptable[e][1]==0 for e in target)
    bad_winding = sum(n==2 and o!=0 for n, o in ptable.values())
    euler = len(set(patch.ravel().tolist()))-len(ptable)+len(patch)
    graph=defaultdict(set)
    for a,b in ptable:
        graph[a].add(b); graph[b].add(a)
    unseen=set(graph); components=0
    while unseen:
        components+=1; stack=[next(iter(unseen))]
        while stack:
            v=stack.pop()
            if v in unseen:
                unseen.remove(v); stack.extend(graph[v])
    bad_links = _patch_vertex_links(patch, {v for e in boundary for v in e}) if len(patch) else 0
    disk = bool(exact_boundary and bad_count==0 and repeated_faces==0 and repeated_vertices==0
                and bad_links==0 and euler==1 and components==1)
    result.update({'patch_faces':len(patch), 'patch_boundary_equals_target':exact_boundary,
                   'target_edges_exactly_two_after_patch':bool(seam_two and len(patch)),
                   'strict_target_closure':bool(result['input_simple_closed_boundary'] and exact_boundary and seam_two),
                   'patch_disk_topology':disk, 'patch_euler_characteristic':int(euler),
                   'patch_connected_components':int(components),
                   'patch_edges_with_invalid_incidence':int(bad_count),
                   'patch_duplicate_faces':int(repeated_faces),
                   'patch_faces_with_repeated_vertex':int(repeated_vertices),
                   'patch_nonmanifold_vertex_links':int(bad_links),
                   'patch_internal_winding_conflicts':int(bad_winding),
                   'seam_winding_opposes_retained':bool(opposing_seam),
                   'orientation_consistent_patch_and_seam':bool(opposing_seam and bad_winding==0)})
    return result


def self_test():
    # Tetrahedron without its base; only [0,1,2] is the true boundary cycle.
    retained=np.array([[0,1,3],[1,2,3],[2,0,3]])
    loop=np.array([0,1,2]); patch=np.array([[0,2,1]])
    q=inspect_local_patch(retained,loop,patch)
    assert q['strict_target_closure'] and q['patch_disk_topology']
    assert q['orientation_consistent_patch_and_seam']
    reverse=inspect_local_patch(retained,loop,patch[:,::-1])
    assert reverse['strict_target_closure'] and not reverse['seam_winding_opposes_retained']
    # Cyclically closing a path through interior vertex 3 is a fake target.
    fake=inspect_local_patch(retained,[0,1,3],[[0,3,1]])
    assert not fake['input_simple_closed_boundary'] and not fake['strict_target_closure']
    assert not inspect_local_patch(retained,[0,1,0,2],patch)['strict_target_closure']
    assert not inspect_local_patch(retained,loop,np.vstack([patch,patch]))['strict_target_closure']
    assert not inspect_local_patch(retained,loop,np.empty((0,3),int))['strict_target_closure']
    return 'strict target cycle, closure, duplicates, winding, disk topology: PASS'


if __name__=='__main__':
    print(self_test())


def patch_disk(f):
 if len(f)==0:return False
 edges=np.vstack([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]); edges=np.sort(edges,axis=1); ue,counts=np.unique(edges,axis=0,return_counts=True); boundary=ue[counts==1]
 if len(boundary)==0:return False
 adj={}
 for a,b in boundary: adj.setdefault(int(a),set()).add(int(b));adj.setdefault(int(b),set()).add(int(a))
 if not adj or any(len(x)!=2 for x in adj.values()):return False
 seen=set(); stack=[next(iter(adj))]
 while stack:
  x=stack.pop()
  if x in seen:continue
  seen.add(x);stack.extend(adj[x]-seen)
 return len(seen)==len(adj) and len(np.unique(f)) - len(ue) + len(f)==1


def edge_counts(f):
 if not len(f):return {}
 edges=np.sort(np.vstack([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1); ue,counts=np.unique(edges,axis=0,return_counts=True)
 return {tuple(map(int,e)):int(n) for e,n in zip(ue,counts)}


def loop_edges(loop):
 return {tuple(sorted((int(a),int(b)))) for a,b in zip(loop,np.roll(loop,-1))}
