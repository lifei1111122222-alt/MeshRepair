from pathlib import Path
import io, json, subprocess, time, hashlib, sys
import numpy as np
import trimesh
from scipy.spatial import cKDTree

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/external_3dcovim'
OUT=ROOT/'experiments/real_scan_external'
sys.path.insert(0,str(ROOT))
NAMES=['amphora','bumpy_sphere','bunny','camel','chicken','cow','dente','dilo','egea','fandisk','femur','foot','liondog','maneki']

def obj_bytes(path):
    return subprocess.run(['7z','e',str(path),'-so'],capture_output=True,check=True).stdout
def load_archive(path):
    return trimesh.load(io.BytesIO(obj_bytes(path)),file_type='obj',process=False)
def boundary_loops(f):
    e=np.sort(np.vstack([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1); u,c=np.unique(e,axis=0,return_counts=True); b=u[c==1]
    adj={}
    for a,d in b: adj.setdefault(int(a),set()).add(int(d)); adj.setdefault(int(d),set()).add(int(a))
    loops=[]; seen=set()
    for s in sorted(adj):
        if s in seen: continue
        cur=[]; x=s; prev=None
        while x not in seen:
            seen.add(x); cur.append(x); nxt=sorted(y for y in adj[x] if y!=prev and y not in seen)
            if not nxt: break
            prev,x=x,nxt[0]
        if len(cur)>=3: loops.append(np.asarray(cur,np.int64))
    return loops
def perimeter(v,loop): return float(np.linalg.norm(v[loop]-np.roll(v[loop],-1,axis=0),axis=1).sum())
def simplify(v,f):
    if len(f)<=15000:return np.asarray(v,float),np.asarray(f,np.int64)
    import fast_simplification
    a,b=fast_simplification.simplify(np.asarray(v,float),np.asarray(f,np.int64),target_count=15000,agg=7.)
    return np.asarray(a,float),np.asarray(b,np.int64)
def sample_local(gt,center,radius):
    pts=np.asarray(gt.vertices,float); d=np.linalg.norm(pts-center,axis=1); ids=np.where(d<=max(radius,1e-6))[0]
    if len(ids)<64: ids=np.argsort(d)[:min(512,len(d))]
    return pts[ids]
def patch_metrics(v,patch,gt,center,radius):
    p=np.asarray(v)[np.unique(np.asarray(patch).reshape(-1))]
    g=sample_local(gt,center,radius)
    if len(p)==0 or len(g)==0:return {'chamfer':None,'hausdorff':None}
    a=cKDTree(g).query(p,k=1)[0]; b=cKDTree(p).query(g,k=1)[0]
    return {'chamfer':float((a.mean()+b.mean())/2),'hausdorff':float(max(a.max(),b.max()))}
def main():
    OUT.mkdir(parents=True,exist_ok=True); (OUT/'cases').mkdir(exist_ok=True)
    from tcmr import integrated_repair as v4
    rows=[]; started_all=time.perf_counter()
    for name in NAMES:
        h=load_archive(DATA/'Meshes-with-holes'/f'{name}_huecos.7z'); gt=load_archive(DATA/'Original-meshes'/f'{name}_original.7z')
        v,f=simplify(h.vertices,h.faces); loops=boundary_loops(f)
        if not loops: rows.append({'sample_id':name,'status':'no_boundary'}); continue
        loop=max(loops,key=lambda x:perimeter(v,x)); center=v[loop].mean(0); rad=float(np.max(np.linalg.norm(v[loop]-center,axis=1))*2.5)
        cue=np.r_[center, max(float(np.linalg.norm(np.ptp(v,axis=0))),1e-6)]
        work=OUT/'cases'/name/'inference'; work.parent.mkdir(parents=True,exist_ok=True)
        t0=time.perf_counter()
        try:
            r=v4.infer(v,f,cue,work,workers=1)
            err=None
        except Exception as e:
            r={}; err=repr(e)
        decisions=r.get('decisions',{})
        row={'sample_id':name,'status':'ok' if err is None else 'failure','error':err,'n_vertices':int(len(v)),'n_faces':int(len(f)),'n_input_boundaries':len(loops),'target_boundary_vertices':int(len(loop)),'runtime_seconds':float(time.perf_counter()-t0),'methods':{}}
        for method in ['prior_locked_pool_with_joint','cgal_library']:
            d=decisions.get(method,{})
            m=d.get('metrics') or {}; pm={}
            pp=d.get('patch_path')
            if pp and Path(pp).exists():
                try:
                    z=np.load(pp); pm=patch_metrics(z['vertices'],z['patch_faces'],gt,center,rad)
                except Exception as e: pm={'metric_error':repr(e)}
            row['methods'][method]={'decision':d.get('decision'),'chosen_method':d.get('chosen_method'),'seam_normal_error':m.get('seam_normal_error'),'quality':m.get('patch_triangle_quality_mean'),'closure':m.get('strict_selected_loop_closure'),'intersection':m.get('introduced_intersection'),'patch_metrics':pm}
        np.savez_compressed(OUT/'cases'/name/'input.npz',vertices=v,faces=f,target_loop=loop,cue=cue)
        rows.append(row); print(name,row['status'],row['runtime_seconds'],flush=True)
    # MeshFix on the same 14 inputs.
    try: import pymeshfix
    except Exception: pymeshfix=None
    for row in rows:
        if row.get('status')!='ok' or pymeshfix is None: continue
        z=np.load(OUT/'cases'/row['sample_id']/'input.npz'); v,f=z['vertices'],z['faces']; t=time.perf_counter()
        try:
            vo,fo=pymeshfix.clean_from_arrays(v,f,verbose=False); vo=np.asarray(vo,float); fo=np.asarray(fo,np.int64); fail=None
            row['meshfix']={'runtime_seconds':float(time.perf_counter()-t),'failed':fail,'watertight':bool(trimesh.Trimesh(vertices=vo,faces=fo,process=False).is_watertight),'n_vertices_out':len(vo),'n_faces_out':len(fo)}
        except Exception as e: row['meshfix']={'runtime_seconds':float(time.perf_counter()-t),'failed':repr(e)}
    def ci(vals):
        vals=np.asarray([x for x in vals if x is not None],float)
        if len(vals)==0:return [None,None]
        rng=np.random.default_rng(20261024); means=np.asarray([vals[rng.integers(0,len(vals),len(vals))].mean() for _ in range(5000)])
        return [float(np.quantile(means,.025)),float(np.quantile(means,.975))]
    summary={}
    for m in ['prior_locked_pool_with_joint','cgal_library']:
        rr=[x for x in rows if x.get('methods',{}).get(m,{}).get('decision')=='accept']; pm=[x['methods'][m]['patch_metrics'].get('chamfer') for x in rr]; hd=[x['methods'][m]['patch_metrics'].get('hausdorff') for x in rr]
        summary[m]={'n':len(rows),'accepted_n':len(rr),'accepted_rate':len(rr)/len(rows) if rows else None,'accepted_ci95':ci([1 if x.get('methods',{}).get(m,{}).get('decision')=='accept' else 0 for x in rows]),'mean_seam':float(np.mean([x['methods'][m]['seam_normal_error'] for x in rr if x['methods'][m]['seam_normal_error'] is not None])) if rr else None,'chamfer_mean':float(np.mean([x for x in pm if x is not None])) if any(x is not None for x in pm) else None,'chamfer_ci95':ci(pm),'hausdorff_mean':float(np.mean([x for x in hd if x is not None])) if any(x is not None for x in hd) else None,'hausdorff_ci95':ci(hd),'runtime_mean':float(np.mean([x['runtime_seconds'] for x in rows])) if rows else None}
    out={'schema':'real_scan_external_benchmark','dataset':'3DCoVim range-sensor meshes with natural holes and complete counterparts','n_cases':len(rows),'source_license':'CC BY 4.0','rows':rows,'summary':summary,'wall_seconds':time.perf_counter()-started_all,'protocol':{'same_damaged_input':True,'cue':'centroid of largest observed boundary loop; supplied equally to TCMR and recorded for CGAL only','gt_used_only_for_scoring':True,'normalization':'fast_simplification QEM target 15000 faces when needed','ci':'case bootstrap 5000 draws'}}
    (OUT/'results.json').write_text(json.dumps(out,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    md=['# Independent natural scan benchmark','',f'3DCoVim range-sensor meshes with natural holes; {len(rows)} source-distinct objects. Complete counterparts are opened only for scoring.','', '|Method|N|Accepted|Rate|Mean seam|Chamfer|Hausdorff|Runtime s|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for m,x in summary.items(): md.append(f"|{m}|{x['n']}|{x['accepted_n']}|{x['accepted_rate']:.3f}|{x['mean_seam'] if x['mean_seam'] is not None else 'NA'}|{x['chamfer_mean'] if x['chamfer_mean'] is not None else 'NA'}|{x['hausdorff_mean'] if x['hausdorff_mean'] is not None else 'NA'}|{x['runtime_mean']:.2f}|")
    md += ['', 'Metrics are local patch-to-ground-truth vertex distances around the selected natural hole. MeshFix is reported per object with global watertightness.']
    (OUT/'results.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
def _read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _save_unified(value):
    path = OUT / 'unified_results.json'
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def unified_protocol():
    sources = []
    for name in NAMES:
        inp = OUT / 'cases' / name / 'input.npz'
        prediction = OUT / 'cases' / name / 'inference' / 'inference.json'
        reference = DATA / 'Original-meshes' / f'{name}_original.7z'
        sources.append({'source_model_id': name, 'input_path': str(inp), 'input_sha256': _file_sha(inp),
                        'prediction_path': str(prediction), 'prediction_sha256': _file_sha(prediction),
                        'reference_path': str(reference), 'reference_sha256': _file_sha(reference)})
    return {'schema': 'natural_scan_unified_cached_rescore', 'n_sources': 14, 'sources': sources,
            'method_configuration': 'Frozen production outputs; no inference or optimization changes',
            'tcmr': 'Score the saved prior_locked_pool_with_joint decision and selected candidate exactly',
            'cgal_pool': 'Cue-nearest boundary among saved shortlisted loop records; two CGAL candidates, explicitly clear native verification, finite minimum recomputed unsigned seam; ties use method name',
            'candidate_universe_limitation': 'Natural scan caches contain at most two prior-shortlisted loops. Record whether the globally cue-nearest observed eligible loop was cached; no new candidate construction',
            'target': 'Saved largest-perimeter input target_loop; comparison uses undirected original rim-edge identities',
            'native': 'Reuse verification JSON, or explicit native status/closure/intersection fields embedded in the frozen candidate record when its JSON is unavailable. Unknown never establishes safety',
            'geometry': 'scripts.evaluate_external_benchmark.geometry_metrics; oriented absolute seam mean/P95, patch q mean/P05, retained RMS and changed-area ratio',
            'reference_roi': 'Center is saved input target_loop vertex mean; radius is 2.5 times maximum input loop vertex distance. GT faces selected by original triangle centroid in closed sphere. No method-dependent ROI or inferred missing-face mask',
            'output_roi': 'Retained face IDs selected once by their ORIGINAL INPUT centroids in the same sphere, evaluated at output coordinates, plus the ENTIRE returned target patch. No clipping of outlying patch faces',
            'reference_endpoint_name': 'Local reconstructed neighborhood surface agreement; ROI contains observed surfaces and is not the exact missing-part reference used for COSEG',
            'surface_samples_per_direction': 4096, 'sampling_seed': 20261025,
            'sampling': 'Area-uniform barycentric samples; one fixed source seed shared across methods; exact closest point on triangle surface',
            'surface_chamfer': '(mean distance output-to-reference + mean distance reference-to-output) / (2*h_in)',
            'sampled_hausdorff': 'Maximum of both directional sample-to-triangle distances / h_in',
            'normal': 'Mean bidirectional acos(abs(normal dot closest-face-normal)); report radians and preserve degrees in raw comparison',
            'normalization': 'h_in is median length of retained-input face edges, exactly as geometry_metrics; no bbox normalization',
            'auxiliary_patch_distance': 'Entire patch samples to reference ROI only: mean/P95/maximum over h_in, with nearest-face normal error; no reverse distance or claim of exact missing-surface recovery',
            'statistics': '20000 source-level paired bootstrap draws, seed20261025; geometry only on actual common correct-safe same-target sources; all14 in outcome denominators; pointwise intervals',
            'scan_failure_breakdown': 'Read existing 24-source by seven-condition report and cached candidate native outcomes; keep confirmed intersections, native unknown/failure, and generation failure separate; report candidate counts and overlapping per-source reasons plus exclusive abstention classes',
            'scan_report_sha256': _file_sha(ROOT / 'experiments/scan_robustness/report.json'),
            'new_inference_runs': 0, 'new_native_verifications': 0, 'new_mesh_exports': 0,
            'timing': 'Scoring time only; no newly measured end-to-end runtime and no reuse of historical combined runtime as fair method timing'}


def _native_record(candidate):
    path = Path(candidate.get('verification_path', ''))
    if path.is_file():
        source = _read_json(path)
        result = {key: source.get(key) for key in ('status', 'strict_selected_loop_closure',
                  'introduced_intersection', 'introduced_intersection_nondegenerate',
                  'introduced_pair_count', 'degenerate_caveat', 'error')}
        result['before_degenerate_faces_n'] = len(source.get('before_degenerate_faces') or [])
        result['after_degenerate_faces_n'] = len(source.get('after_degenerate_faces') or [])
        result['evidence_source'] = str(path)
        result['evidence_sha256'] = _file_sha(path)
        return result
    return {'status': candidate.get('verification_status', 'missing'),
            'strict_selected_loop_closure': candidate.get('strict_selected_loop_closure'),
            'introduced_intersection': candidate.get('introduced_intersection'),
            'evidence_source': 'explicit_native_fields_in_saved_candidate'}


def _native_clear(audit):
    return (audit.get('status') == 'success' and audit.get('strict_selected_loop_closure') is True
            and audit.get('introduced_intersection') is False)


def _rim_edges(loop):
    return {tuple(sorted((int(a), int(b)))) for a, b in zip(loop, np.roll(loop, -1))}


def _boundary_edges(faces):
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    return {tuple(map(int, edge)) for edge in unique[counts == 1]}


def _source_interval(first, second=None):
    first = np.asarray(first, float)
    delta = first if second is None else first - np.asarray(second, float)
    if not len(delta):
        return {'n': 0, 'mean': None, 'ci95': None}
    rng = np.random.default_rng(20261025)
    draws = delta[rng.integers(0, len(delta), size=(20000, len(delta)))].mean(axis=1)
    result = {'n': len(delta), 'mean': float(delta.mean()), 'ci95': np.quantile(draws, [.025, .975]).tolist()}
    if second is not None:
        result.update(first_mean=float(first.mean()), second_mean=float(np.mean(second)))
    return result


def _nested(item, key):
    for part in key.split('.'):
        item = item[part]
    return item


def _scan_failure_breakdown():
    folder = ROOT / 'experiments/scan_robustness'
    report, protocol = _read_json(folder / 'report.json'), _read_json(folder / 'protocol.json')
    cases = {row['sample_id']: row for row in protocol['cases']}
    rows = []
    for scored in report['rows']:
        case = cases[scored['sample_id']]
        row = {k: scored[k] for k in ('sample_id', 'source_model_id', 'category', 'condition')}
        row['methods'] = {}
        for name in ('tcmr', 'cgal_library'):
            pred = _read_json(folder / 'predictions' / f"{case['index']:03d}_{name}.json")
            decision = pred.get('decisions', {}).get('prior_locked_pool_with_joint', {})
            locked = decision.get('locked_loop_id')
            if locked is None:
                locked = decision.get('selected_loop_id')
            loops = [loop for loop in pred.get('loops', []) if loop.get('loop_id') == locked]
            if locked is None and len(pred.get('loops', [])) == 1:
                loops = pred['loops']
            states = []
            for loop in loops:
                for candidate in loop.get('candidates', []):
                    audit = _native_record(candidate)
                    if audit.get('status') == 'success' and audit.get('introduced_intersection') is True:
                        reason = 'confirmed_intersection'
                    elif candidate.get('status') != 'success':
                        reason = 'generation_failure'
                    elif audit.get('status') != 'success' or audit.get('introduced_intersection') is None:
                        reason = 'native_unknown_or_failure'
                    elif audit.get('strict_selected_loop_closure') is not True:
                        reason = 'closure_failure'
                    elif candidate.get('safe_eligible') is True:
                        reason = 'safe_eligible'
                    else:
                        reason = 'other_ineligible'
                    states.append({'method': candidate['method'], 'reason': reason, 'native': audit,
                                   'generation_status': candidate.get('status'), 'error': candidate.get('error')})
            counts = {reason: sum(s['reason'] == reason for s in states) for reason in (
                'confirmed_intersection', 'generation_failure', 'native_unknown_or_failure', 'closure_failure', 'safe_eligible', 'other_ineligible')}
            accepted = scored[name].get('accepted', scored[name].get('decision') == 'accept')
            if accepted:
                outcome = 'correct_safe' if scored[name].get('correct_safe') is True else 'accepted_unsuccessful'
            elif not states:
                outcome = 'abstain_missing_selected_candidate_records'
            elif counts['confirmed_intersection'] and (counts['native_unknown_or_failure'] or counts['generation_failure']):
                outcome = 'abstain_mixed_intersection_and_unresolved'
            elif counts['confirmed_intersection']:
                outcome = 'abstain_confirmed_intersection_without_unresolved'
            elif counts['native_unknown_or_failure']:
                outcome = 'abstain_native_unknown_or_failure'
            elif counts['generation_failure'] == len(states):
                outcome = 'abstain_generation_failure_only'
            else:
                outcome = 'abstain_other'
            row['methods'][name] = {'accepted': bool(accepted), 'correct_safe': scored[name].get('correct_safe') is True,
                'abstain': not accepted, 'locked_loop_id': locked, 'outcome': outcome, 'candidate_reason_counts': counts, 'candidates': states}
        rows.append(row)
    summary = {}
    for condition in [item['name'] for item in protocol['conditions']]:
        group = [r for r in rows if r['condition'] == condition]
        summary[condition] = {'source_n': len(group), 'methods': {}}
        for name in ('tcmr', 'cgal_library'):
            items = [r['methods'][name] for r in group]
            unknown = [c for item in items for c in item['candidates'] if c['reason'] == 'native_unknown_or_failure']
            summary[condition]['methods'][name] = {
                'correct_safe_n': sum(x['correct_safe'] for x in items), 'accepted_n': sum(x['accepted'] for x in items),
                'abstain_n': sum(x['abstain'] for x in items),
                'exclusive_outcomes': {reason: sum(x['outcome'] == reason for x in items) for reason in sorted({x['outcome'] for x in items})},
                'candidate_reason_totals': {reason: sum(x['candidate_reason_counts'][reason] for x in items) for reason in items[0]['candidate_reason_counts']},
                'unknown_native_status_counts': {status: sum(c['native'].get('status') == status for c in unknown)
                    for status in sorted({c['native'].get('status', 'missing') for c in unknown})},
                'unknown_candidates_with_degenerate_faces_n': sum(c['native'].get('after_degenerate_faces_n', 0) > 0 for c in unknown),
                'overlapping_abstention_source_reasons': {reason: [r['source_model_id'] for r in group
                    if r['methods'][name]['abstain'] and r['methods'][name]['candidate_reason_counts'][reason] > 0]
                    for reason in ('confirmed_intersection', 'native_unknown_or_failure', 'generation_failure')}}
    return {'n_sources': len({r['source_model_id'] for r in rows}), 'n_repeated_conditions': len(rows),
            'candidate_counts_are_not_independent_sources': True, 'by_condition': summary, 'rows': rows}


def rescore_unified(prepare_only=False):
    from scripts.evaluate_geometry_comparison import _seam_normal_error
    from scripts.evaluate_external_benchmark import geometry_metrics
    from scripts.evaluate_reference_surface import compare_surface, sample_surface, surface, triangle_info, closest
    from tcmr.boundary import extract_boundary_loops
    from scripts.local_repair_topology import inspect_target_cycle
    protocol = unified_protocol()
    destination = OUT / 'unified_results.json'
    if destination.exists():
        prior = _read_json(destination)
        if prior['protocol'] != protocol:
            raise ValueError('Prepared unified scoring protocol or source files changed')
        if prior.get('complete'):
            print(json.dumps({'complete': True, 'reused_existing': True, 'output': str(destination)}), flush=True)
            return
    else:
        _save_unified({'complete': False, 'status': 'protocol_prepared', 'protocol': protocol, 'rows': []})
    if prepare_only:
        print(json.dumps({'prepared': True, 'n_sources': 14, 'output': str(destination)}), flush=True)
        return
    rows = []
    started = time.perf_counter()
    for index, source in enumerate(protocol['sources']):
        with np.load(source['input_path'], allow_pickle=False) as data:
            before, retained, target, cue = data['vertices'], data['faces'], data['target_loop'], data['cue']
        prediction = _read_json(source['prediction_path'])
        center = before[target].mean(axis=0)
        radius = float(2.5 * np.linalg.norm(before[target] - center, axis=1).max())
        retained_ids = np.flatnonzero(np.linalg.norm(before[retained].mean(axis=1) - center, axis=1) <= radius)
        reference = load_archive(Path(source['reference_path']))
        gv, gf = np.asarray(reference.vertices, float), np.asarray(reference.faces, np.int64)
        reference_ids = np.flatnonzero(np.linalg.norm(gv[gf].mean(axis=1) - center, axis=1) <= radius)
        if not len(reference_ids) or not len(retained_ids) or radius <= 0:
            raise ValueError('Empty prespecified local ROI for ' + source['source_model_id'])
        ref_faces = gf[reference_ids]
        ref_area, ref_normals = triangle_info(gv, ref_faces)
        original_loops = [loop for loop in extract_boundary_loops(before, retained)
                          if inspect_target_cycle(retained, loop)['input_simple_closed_boundary']]
        nearest_input_loop = min(original_loops, key=lambda loop: (np.linalg.norm(before[loop].mean(axis=0) - cue[:3]), tuple(loop)))
        choices = []
        for record in prediction['loops']:
            with np.load(record['input_path'], allow_pickle=False) as data:
                if not np.array_equal(data['vertices'], before) or not np.array_equal(data['faces'], retained):
                    raise ValueError('Cached loop input differs from natural-scan source')
                loop = data['selected_loop'].copy()
            choices.append((float(np.linalg.norm(before[loop].mean(axis=0) - cue[:3])), record['loop_id'], record, loop))
        nearest = min(choices, key=lambda x: (x[0], x[1]))
        decision = prediction['decisions']['prior_locked_pool_with_joint']
        selected = next((x for x in choices if x[1] == decision.get('selected_loop_id')), None)
        tcmr_candidate = next((c for c in selected[2]['candidates'] if c['method'] == decision.get('chosen_method')), None) if selected else None
        cgal_scored = []
        for candidate in nearest[2]['candidates']:
            if candidate['method'] not in ('cgal_refine_fair', 'cgal_polyline'):
                continue
            audit = _native_record(candidate)
            if not _native_clear(audit) or not candidate.get('patch_path'):
                continue
            with np.load(candidate['patch_path'], allow_pickle=False) as data:
                score = _seam_normal_error(data['vertices'], retained, data['patch_faces'], nearest[3])
            if score is not None and np.isfinite(score):
                cgal_scored.append((score, candidate['method'], candidate))
        cgal_candidate = min(cgal_scored, key=lambda x: (x[0], x[1]))[2] if cgal_scored else None
        seed = 20261025 + 1009 * index
        row = {'source_model_id': source['source_model_id'], 'reference': {
            'center': center.tolist(), 'radius': radius, 'retained_face_ids': retained_ids.tolist(),
            'reference_face_ids': reference_ids.tolist(), 'retained_roi_faces_n': len(retained_ids),
            'reference_roi_faces_n': len(reference_ids), 'reference_roi_area': float(ref_area.sum()),
            'observed_input_roi_area': float(triangle_info(before, retained[retained_ids])[0].sum()),
            'sample_seed': seed, 'reference_vertices_n': len(gv), 'reference_faces_n': len(gf)},
            'globally_cue_nearest_loop_cached': any(_rim_edges(x[3]) == _rim_edges(nearest_input_loop) for x in choices),
            'methods': {}}
        for name, record, candidate, accepted in (
            ('tcmr', selected, tcmr_candidate, decision.get('decision') == 'accept'),
            ('cgal_pool', nearest, cgal_candidate, cgal_candidate is not None)):
            result = {'accepted': accepted, 'correct_safe': False, 'selected_target': False,
                      'selected_loop_id': record[1] if record else None, 'chosen_method': candidate['method'] if candidate else None}
            if accepted and record and candidate:
                audit = _native_record(candidate)
                with np.load(candidate['patch_path'], allow_pickle=False) as data:
                    vertices, patch = data['vertices'], data['patch_faces']
                geom = geometry_metrics(before, retained, vertices, patch, record[3])
                selected_target = _rim_edges(record[3]) == _rim_edges(target)
                unrelated = _boundary_edges(retained) - _rim_edges(record[3])
                preserved = unrelated <= _boundary_edges(np.vstack([retained, patch]))
                valid = bool(_native_clear(audit) and geom.get('geometry_no_new_topology_defect') is True and preserved)
                result.update(selected_target=selected_target, correct_safe=valid and selected_target,
                              valid_selected_loop=valid, unrelated_boundaries_preserved=preserved,
                              native=audit, geometry=geom, patch_path=candidate['patch_path'], patch_faces_n=len(patch),
                              selection_seam_unoriented=_seam_normal_error(vertices, retained, patch, record[3]))
                if selected_target:
                    h = geom['median_input_edge']
                    local_faces = np.vstack([retained[retained_ids], patch])
                    local = compare_surface(vertices, local_faces, gv, ref_faces, h, 4096, seed)
                    local['normal_angle_rad'] = float(np.radians(local['normal_angle_degrees']))
                    points, normals, patch_area = sample_surface(vertices, patch, 4096, seed)
                    distances, nearest_normals = closest(surface(gv, ref_faces), points, ref_normals)
                    d = distances / h
                    result.update(local_reference=local, patch_to_reference={
                        'mean_over_h': float(d.mean()), 'p95_over_h': float(np.percentile(d, 95)), 'maximum_over_h': float(d.max()),
                        'normal_angle_rad': float(np.arccos(np.clip(np.abs(np.einsum('ij,ij->i', normals, nearest_normals)), 0, 1)).mean()),
                        'patch_area': patch_area, 'samples': 4096})
            row['methods'][name] = result
        rows.append(row)
        _save_unified({'complete': False, 'status': 'scoring', 'protocol': protocol, 'rows': rows})
        print(json.dumps({'scored_sources': len(rows), 'source': source['source_model_id'],
                          'correct_safe': {k: v['correct_safe'] for k, v in row['methods'].items()}}), flush=True)
    common = [r for r in rows if all(r['methods'][m]['correct_safe'] for m in ('tcmr', 'cgal_pool'))]
    metrics = ('geometry.seam_normal_absolute_mean', 'geometry.seam_normal_absolute_p95', 'geometry.triangle_quality.mean',
               'geometry.triangle_quality.p05', 'geometry.retained_displacement_rms_over_edge', 'geometry.changed_surface_area_ratio',
               'local_reference.surface_chamfer_l1_over_h', 'local_reference.sampled_hausdorff_over_h',
               'local_reference.normal_angle_rad', 'patch_to_reference.mean_over_h', 'patch_to_reference.p95_over_h',
               'patch_to_reference.maximum_over_h')
    summary = {'source_n': len(rows), 'common_correct_safe_n': len(common), 'common_source_ids': [r['source_model_id'] for r in common],
               'methods': {}, 'paired_tcmr_minus_cgal': {}}
    for name in ('tcmr', 'cgal_pool'):
        items = [r['methods'][name] for r in rows]
        summary['methods'][name] = {'accepted_n': sum(x['accepted'] for x in items), 'correct_safe_n': sum(x['correct_safe'] for x in items),
            'abstain_n': sum(not x['accepted'] for x in items), 'wrong_accepted_n': sum(x['accepted'] and not x['correct_safe'] for x in items),
            'correct_safe_rate': _source_interval([float(x['correct_safe']) for x in items]),
            'common_geometry': {key: _source_interval([_nested(r['methods'][name], key) for r in common]) for key in metrics}}
    summary['paired_tcmr_minus_cgal'] = {key: _source_interval([_nested(r['methods']['tcmr'], key) for r in common],
        [_nested(r['methods']['cgal_pool'], key) for r in common]) for key in metrics}
    summary['paired_correct_safe_all_sources'] = _source_interval([float(r['methods']['tcmr']['correct_safe']) for r in rows],
        [float(r['methods']['cgal_pool']['correct_safe']) for r in rows])
    _save_unified({'complete': True, 'status': 'complete', 'protocol': protocol, 'rows': rows, 'summary': summary,
                  'scan_failure_breakdown': _scan_failure_breakdown(), 'scoring_wall_seconds': time.perf_counter() - started})
    print(json.dumps({'complete': True, 'common_correct_safe_n': len(common), 'output': str(destination)}), flush=True)


def _save_independent(value):
    path = OUT / 'independent_cgal_results.json'
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def _exact_source_rate(values):
    from scipy.stats import beta
    n, k = len(values), int(sum(values))
    return {'n': n, 'success_n': k, 'mean': k / n if n else None,
            'ci95': [float(beta.ppf(.025, k, n-k+1)) if k else 0.,
                     float(beta.ppf(.975, k+1, n-k)) if k < n else 1.],
            'method': 'Clopper-Pearson exact binomial 95% interval'}


def _independent_cgal_geometry(before, retained, target, loop, candidate, reference_record, source, index):
    """Identical saved natural-scan metrics; target labels are opened for scoring only."""
    from scripts.evaluate_geometry_comparison import _seam_normal_error
    from scripts.evaluate_external_benchmark import geometry_metrics
    from scripts.evaluate_reference_surface import compare_surface, sample_surface, surface, triangle_info, closest
    result = {'accepted': candidate is not None, 'correct_safe': False, 'selected_target': False,
              'chosen_method': candidate['method'] if candidate else None}
    if candidate is None:
        return result
    audit = _native_record(candidate)
    with np.load(candidate['patch_path'], allow_pickle=False) as data:
        vertices, patch = data['vertices'], data['patch_faces']
    geom = geometry_metrics(before, retained, vertices, patch, loop)
    selected_target = _rim_edges(loop) == _rim_edges(target)
    unrelated = _boundary_edges(retained) - _rim_edges(loop)
    preserved = unrelated <= _boundary_edges(np.vstack([retained, patch]))
    valid = bool(_native_clear(audit) and geom.get('geometry_no_new_topology_defect') is True and preserved)
    result.update(selected_target=selected_target, correct_safe=valid and selected_target,
                  valid_selected_loop=valid, unrelated_boundaries_preserved=preserved,
                  native=audit, geometry=geom, patch_path=candidate['patch_path'], patch_faces_n=len(patch),
                  selection_seam_unoriented=_seam_normal_error(vertices, retained, patch, loop))
    if selected_target:
        reference = load_archive(Path(source['reference_path']))
        gv, gf = np.asarray(reference.vertices, float), np.asarray(reference.faces, np.int64)
        retained_ids = np.asarray(reference_record['retained_face_ids'], np.int64)
        reference_ids = np.asarray(reference_record['reference_face_ids'], np.int64)
        ref_faces = gf[reference_ids]
        _, ref_normals = triangle_info(gv, ref_faces)
        h, seed = geom['median_input_edge'], 20261025 + 1009 * index
        local = compare_surface(vertices, np.vstack([retained[retained_ids], patch]), gv, ref_faces, h, 4096, seed)
        local['normal_angle_rad'] = float(np.radians(local['normal_angle_degrees']))
        points, normals, patch_area = sample_surface(vertices, patch, 4096, seed)
        distances, nearest_normals = closest(surface(gv, ref_faces), points, ref_normals)
        d = distances / h
        result.update(local_reference=local, patch_to_reference={
            'mean_over_h': float(d.mean()), 'p95_over_h': float(np.percentile(d, 95)),
            'maximum_over_h': float(d.max()),
            'normal_angle_rad': float(np.arccos(np.clip(np.abs(np.einsum('ij,ij->i', normals, nearest_normals)), 0, 1)).mean()),
            'patch_area': patch_area, 'samples': 4096})
    return result


def independent_cgal():
    """Cue-nearest CGAL on ALL eligible input boundaries, independent of the learned shortlist."""
    from tcmr import integrated_repair as method
    from tcmr.boundary import extract_boundary_loops
    from scripts.local_repair_topology import inspect_target_cycle
    from scripts.evaluate_geometry_comparison import _seam_normal_error
    saved_path = OUT / 'unified_results.json'
    saved = _read_json(saved_path)
    if not saved.get('complete'):
        raise ValueError('Complete unified natural-scan scoring is required first')
    protocol = dict(saved['protocol'])
    protocol.update(schema='natural_scan_independent_cue_nearest_cgal',
        cgal_pool='All observed eligible simple boundary loops from original input; cue-centroid Euclidean distance then original extracted loop ID; lock identity before generation; safe CGAL refine/fair + polyline pool; minimum finite unsigned seam then method name',
        candidate_universe_limitation='None for the independent CGAL comparator: all extracted loops are inspected; no learned score or shortlist is used',
        eligibility='inspect_target_cycle(input_faces, loop)[input_simple_closed_boundary]; matches the independent COSEG CGAL comparator',
        cached_metric_source=str(saved_path.resolve()), cached_metric_source_sha256=_file_sha(saved_path),
        endpoint_rate_interval='Clopper-Pearson exact 95% intervals for individual correct-safe rates; paired rate difference uses 20000 source bootstrap draws',
        reuse='Original TCMR decisions and metrics unchanged. Reuse exact-rim CGAL candidates and metrics when available; generate two CGAL candidates only for missing selected loops; score using identical saved ROI and samples',
        new_inference_runs=0, new_native_verifications=None, new_mesh_exports=None,
        timing='No fair end-to-end timing claimed: cached geometry for 13 sources and new two-candidate worker times for a previously missing loop; scoring time recorded separately')
    destination = OUT / 'independent_cgal_results.json'
    if destination.exists():
        prior = _read_json(destination)
        if prior['protocol'] != protocol:
            raise ValueError('Independent CGAL protocol or source hashes changed')
        if prior.get('complete'):
            print(json.dumps({'complete': True, 'reused_existing': True, 'output': str(destination)}), flush=True)
            return
    rows, new_candidates = [], 0
    started = time.perf_counter()
    saved_rows = {r['source_model_id']: r for r in saved['rows']}
    for index, source in enumerate(protocol['sources']):
        name = source['source_model_id']
        with np.load(source['input_path'], allow_pickle=False) as data:
            before, retained, target, cue = data['vertices'], data['faces'], data['target_loop'], data['cue']
        loops = extract_boundary_loops(before, retained)
        choices = []
        inventory = []
        for loop_id, loop in enumerate(loops):
            simple = inspect_target_cycle(retained, loop)['input_simple_closed_boundary'] is True
            distance = float(np.linalg.norm(before[loop].mean(axis=0) - cue[:3]))
            inventory.append({'loop_id': loop_id, 'vertices_n': len(loop), 'eligible_simple_cycle': simple,
                              'cue_centroid_distance': distance})
            if simple:
                choices.append((distance, loop_id, loop))
        selected = min(choices, key=lambda x: (x[0], x[1])) if choices else None
        candidates, origin, scored = [], 'no_eligible_input_boundary', []
        cached_loop_id = None
        if selected:
            prediction = _read_json(source['prediction_path'])
            for record in prediction['loops']:
                with np.load(record['input_path'], allow_pickle=False) as data:
                    if not np.array_equal(data['vertices'], before) or not np.array_equal(data['faces'], retained):
                        raise ValueError('Cached loop differs from original input')
                    cached_loop = data['selected_loop']
                if _rim_edges(cached_loop) == _rim_edges(selected[2]):
                    candidates = [c for c in record['candidates'] if c['method'] in ('cgal_refine_fair', 'cgal_polyline')]
                    cached_loop_id, origin = record['loop_id'], 'original_exact_rim_cache'
                    break
            if not candidates:
                work = OUT / 'cases' / name / 'independent_cgal'
                work.mkdir(exist_ok=True)
                inp = work / 'damaged_input.npz'
                if not inp.exists():
                    np.savez_compressed(inp, vertices=before, faces=retained, selected_loop=selected[2])
                with np.load(inp, allow_pickle=False) as data:
                    if not all(np.array_equal(data[k], a) for k, a in (
                            ('vertices', before), ('faces', retained), ('selected_loop', selected[2]))):
                        raise ValueError('Independent selected-loop input changed')
                key = method.content_key(before, retained, selected[2])
                for candidate_name in ('cgal_refine_fair', 'cgal_polyline'):
                    result_path = work / (candidate_name + '.json')
                    if result_path.exists():
                        candidate = _read_json(result_path)
                        if candidate.get('input_geometry_key') != key:
                            raise ValueError('Independent CGAL candidate input changed')
                    else:
                        candidate = method._run_candidate({'method': candidate_name, 'input_path': str(inp.resolve()),
                            'input_geometry_key': key}, result_path)
                        new_candidates += 1
                    candidates.append(candidate)
                origin = 'independent_two_candidate_generation'
            for candidate in candidates:
                audit = _native_record(candidate)
                if not _native_clear(audit) or not candidate.get('patch_path'):
                    continue
                with np.load(candidate['patch_path'], allow_pickle=False) as data:
                    score = _seam_normal_error(data['vertices'], retained, data['patch_faces'], selected[2])
                if score is not None and np.isfinite(score):
                    scored.append((score, candidate['method'], candidate))
        chosen = min(scored, key=lambda x: (x[0], x[1]))[2] if scored else None
        old_row = saved_rows[name]
        old_cgal = old_row['methods']['cgal_pool']
        if chosen is not None and old_cgal.get('patch_path') == chosen['patch_path']:
            cgal = dict(old_cgal)
            metric_origin = 'exact_same_cached_candidate_unified_metrics'
        elif selected:
            cgal = _independent_cgal_geometry(before, retained, target, selected[2], chosen, old_row['reference'], source, index)
            metric_origin = 'same_protocol_recomputed_metrics'
        else:
            cgal = {'accepted': False, 'correct_safe': False, 'selected_target': False, 'chosen_method': None}
            metric_origin = 'no_output'
        cgal['selected_loop_id'] = selected[1] if selected else None
        cgal.update(candidate_origin=origin, metric_origin=metric_origin, cached_loop_id=cached_loop_id)
        row = {'source_model_id': name, 'reference': old_row['reference'],
            'boundary_inventory': inventory, 'eligible_input_loops_n': len(choices),
            'selected_input_loop_id': selected[1] if selected else None,
            'selected_input_loop': selected[2].tolist() if selected else None,
            'selected_input_loop_distance': selected[0] if selected else None,
            'selected_loop_exact_cache_available': cached_loop_id is not None,
            'cgal_candidates': candidates, 'methods': {'tcmr': old_row['methods']['tcmr'], 'cgal_pool': cgal}}
        rows.append(row)
        _save_independent({'complete': False, 'status': 'scoring', 'protocol': protocol, 'rows': rows})
        print(json.dumps({'scored_sources': len(rows), 'source': name, 'candidate_origin': origin,
                          'correct_safe': {k: v['correct_safe'] for k, v in row['methods'].items()}}), flush=True)
    common = [r for r in rows if all(r['methods'][m]['correct_safe'] for m in ('tcmr', 'cgal_pool'))]
    metrics = ('geometry.seam_normal_absolute_mean', 'geometry.seam_normal_absolute_p95',
        'geometry.triangle_quality.mean', 'geometry.triangle_quality.p05',
        'geometry.retained_displacement_rms_over_edge', 'geometry.changed_surface_area_ratio',
        'local_reference.surface_chamfer_l1_over_h', 'local_reference.sampled_hausdorff_over_h',
        'local_reference.normal_angle_rad', 'patch_to_reference.mean_over_h',
        'patch_to_reference.p95_over_h', 'patch_to_reference.maximum_over_h')
    summary = {'source_n': len(rows), 'common_correct_safe_n': len(common),
        'common_source_ids': [r['source_model_id'] for r in common], 'methods': {}, 'paired_tcmr_minus_cgal': {}}
    for name in ('tcmr', 'cgal_pool'):
        items = [r['methods'][name] for r in rows]
        summary['methods'][name] = {'accepted_n': sum(x['accepted'] for x in items),
            'correct_safe_n': sum(x['correct_safe'] for x in items),
            'abstain_n': sum(not x['accepted'] for x in items),
            'wrong_accepted_n': sum(x['accepted'] and not x['correct_safe'] for x in items),
            'correct_safe_rate': _exact_source_rate([float(x['correct_safe']) for x in items]),
            'common_geometry': {key: _source_interval([_nested(r['methods'][name], key) for r in common]) for key in metrics}}
    summary['paired_tcmr_minus_cgal'] = {key: _source_interval(
        [_nested(r['methods']['tcmr'], key) for r in common],
        [_nested(r['methods']['cgal_pool'], key) for r in common]) for key in metrics}
    summary['paired_correct_safe_all_sources'] = _source_interval(
        [float(r['methods']['tcmr']['correct_safe']) for r in rows],
        [float(r['methods']['cgal_pool']['correct_safe']) for r in rows])
    _save_independent({'complete': True, 'status': 'complete', 'protocol': protocol, 'rows': rows,
        'summary': summary, 'new_candidate_generation_n': new_candidates,
        'cached_selected_loop_sources_n': sum(r['selected_loop_exact_cache_available'] for r in rows),
        'scoring_and_missing_candidate_wall_seconds': time.perf_counter() - started})
    print(json.dumps({'complete': True, 'common_correct_safe_n': len(common), 'new_candidates': new_candidates,
                      'output': str(destination)}), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-rescore', action='store_true')
    parser.add_argument('--rescore', action='store_true')
    parser.add_argument('--independent-cgal', action='store_true')
    args = parser.parse_args()
    if args.independent_cgal:
        independent_cgal()
    elif args.prepare_rescore or args.rescore:
        rescore_unified(prepare_only=args.prepare_rescore)
    else:
        main()
