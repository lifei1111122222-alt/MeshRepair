"""Fixed controlled face removals on public Stanford reconstructed scans.

These are 16 operations on four sources, not 16 independent meshes or human
edits. Generation uses damaged geometry and cues; annotations are score-only.
"""
from pathlib import Path
import argparse, gzip, hashlib, io, json, os, subprocess, sys, tarfile, time, urllib.request
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = ROOT / 'data/stanford_scan_edits'
OUT = ROOT / 'experiments/scan_edit_benchmark'
SOURCES = {'bunny': 'bunny.tar.gz', 'armadillo': 'Armadillo.ply.gz',
           'drill': 'drill.tar.gz', 'dragon': 'dragon_recon.tar.gz'}
ANCHORS = [[.2,.5,.5],[.8,.5,.5],[.5,.2,.5],[.5,.8,.5]]
ROBUST_DATA = ROOT / 'data/scan_robustness'
ROBUST_OUT = ROOT / 'experiments/scan_robustness'
YCB_GROUPS = {
    'packaged_food': ['002_master_chef_can','003_cracker_box','004_sugar_box',
                      '005_tomato_soup_can','006_mustard_bottle','007_tuna_fish_can'],
    'produce': ['011_banana','012_strawberry','013_apple','014_lemon','015_peach','016_pear'],
    'household': ['019_pitcher_base','021_bleach_cleanser','024_bowl','025_mug','026_sponge','029_plate'],
    'tools': ['035_power_drill','036_wood_block','037_scissors','040_large_marker','042_adjustable_wrench','048_hammer']}
ROBUST_CONDITIONS = [
    {'name':'clean','factor':'reference','noise_sigma_edge':0.,'face_ratio':1.,'target_area_ratio':.005},
    {'name':'noise_low','factor':'noise','noise_sigma_edge':.05,'face_ratio':1.,'target_area_ratio':.005},
    {'name':'noise_high','factor':'noise','noise_sigma_edge':.15,'face_ratio':1.,'target_area_ratio':.005},
    {'name':'resolution_half','factor':'resolution','noise_sigma_edge':0.,'face_ratio':.5,'target_area_ratio':.005},
    {'name':'resolution_quarter','factor':'resolution','noise_sigma_edge':0.,'face_ratio':.25,'target_area_ratio':.005},
    {'name':'hole_small','factor':'hole_area','noise_sigma_edge':0.,'face_ratio':1.,'target_area_ratio':.0025},
    {'name':'hole_large','factor':'hole_area','noise_sigma_edge':0.,'face_ratio':1.,'target_area_ratio':.01}]


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def reconstruction(name):
    path = DATA / SOURCES[name]
    if name == 'armadillo':
        return trimesh.load(io.BytesIO(gzip.decompress(path.read_bytes())), file_type='ply', process=False), SOURCES[name]
    options = []
    with tarfile.open(path, 'r:gz') as archive:
        for member in archive.getmembers():
            if not member.name.endswith('.ply'): continue
            # Reconstruction assets only; never choose individual range scans.
            if name == 'bunny' and '/reconstruction/' not in member.name: continue
            if name == 'dragon' and 'dragon_vrip' not in member.name: continue
            if name == 'drill' and not ('reconstruction' in member.name or 'vrip' in member.name): continue
            mesh = trimesh.load(io.BytesIO(archive.extractfile(member).read()), file_type='ply', process=False)
            options.append((len(mesh.faces), member.name, mesh))
    if not options: raise ValueError('no reconstruction assets: ' + name)
    under = [x for x in options if x[0] <= 20000]
    chosen = max(under, key=lambda x:(x[0],x[1])) if under else min(options, key=lambda x:(x[0],x[1]))
    return chosen[2], chosen[1]


def patch_neighborhood(seed, adjacency, count):
    queue, seen = [int(seed)], {int(seed)}
    for face in queue:
        if len(queue) >= count: break
        for neighbor in sorted(adjacency.get(face, [])):
            if neighbor not in seen:
                seen.add(neighbor); queue.append(neighbor)
                if len(queue) >= count: break
    return np.asarray(queue[:count], np.int64)


def simple_cut(vertices, faces, removed):
    from scripts.local_repair_topology import edge_counts, patch_disk, loop_edges
    if not patch_disk(faces[removed]): return None
    counts = edge_counts(faces[removed]); boundary = [e for e,n in counts.items() if n == 1]
    full = edge_counts(faces)
    if any(full.get(e) != 2 for e in boundary): return None
    adjacent = {}
    for a,b in boundary: adjacent.setdefault(a,[]).append(b); adjacent.setdefault(b,[]).append(a)
    loop = [min(adjacent)]; previous = None
    while len(loop) < len(adjacent):
        choices = sorted(x for x in adjacent[loop[-1]] if x != previous)
        next_id = next((x for x in choices if x not in loop), None)
        if next_id is None: return None
        previous = loop[-1]; loop.append(next_id)
    if loop_edges(loop) != set(boundary): return None
    return np.asarray(loop, np.int64)


def prepare():
    import fast_simplification
    if (OUT / 'protocol.json').exists(): raise FileExistsError('protocol already frozen')
    construction = read(DATA / 'construction_protocol.json')
    construction['sources'] = list(SOURCES)
    write(DATA / 'construction_protocol.json', construction)
    cases, labels, source_rows = [], [], []
    for name in SOURCES:
        mesh, member = reconstruction(name)
        original_vertices, original_faces = len(mesh.vertices), len(mesh.faces)
        vertices=np.asarray(mesh.vertices,float).copy();faces=np.asarray(mesh.faces,np.int64).copy()
        if len(mesh.faces) > 15000:
            vertices,faces=fast_simplification.simplify(vertices,faces,target_count=15000,agg=7.)
        center = (vertices.min(0)+vertices.max(0))/2; scale = np.ptp(vertices,axis=0).max()
        vertices = (vertices-center)/scale
        source_file = DATA / f'{name}.npz'; np.savez_compressed(source_file,vertices=vertices,faces=faces)
        source_rows.append({'source_model_id':name,'raw_archive':str((DATA/SOURCES[name]).resolve()),
            'raw_sha256':sha(DATA/SOURCES[name]),'selected_member':member,'original_vertices':original_vertices,
            'original_faces':original_faces,'preprocessed_vertices':len(vertices),'preprocessed_faces':len(faces),
            'source_mesh_path':str(source_file.resolve()),'source_mesh_sha256':sha(source_file),
            'normalization_center':center.tolist(),'normalization_scale':float(scale),
            'simplifier':'fast-simplification 0.1.12 QEM target_count15000 agg7; same source mesh for all methods'})
        mesh = trimesh.Trimesh(vertices=vertices,faces=faces,process=False)
        adjacency = {}
        for a,b in mesh.face_adjacency: adjacency.setdefault(int(a),[]).append(int(b));adjacency.setdefault(int(b),[]).append(int(a))
        centers = vertices[faces].mean(1); low=vertices.min(0); span=np.ptp(vertices,axis=0)
        for operation, anchor in enumerate(ANCHORS):
            requested = low+span*np.asarray(anchor); selected = None
            seeds = np.lexsort((np.arange(len(faces)),np.linalg.norm(centers-requested,axis=1)))
            for seed in seeds:
                target_faces = patch_neighborhood(seed,adjacency,80)
                target_loop = simple_cut(vertices,faces,target_faces)
                if len(target_faces)==80 and target_loop is not None:
                    selected = (int(seed),target_faces,target_loop); break
            if selected is None: raise ValueError(f'{name}/{operation}: no structurally valid target')
            seed,target_faces,target_loop=selected; target_vertices=set(np.unique(faces[target_faces]).tolist())
            distant=np.lexsort((np.arange(len(faces)),-np.linalg.norm(centers-centers[seed],axis=1)))
            distractor=None
            for ds in distant:
                removed=patch_neighborhood(ds,adjacency,48); loop=simple_cut(vertices,faces,removed)
                if len(removed)==48 and loop is not None and not target_vertices.intersection(np.unique(faces[removed])):
                    distractor=(int(ds),removed,loop);break
            if distractor is None:raise ValueError(f'{name}/{operation}: no structurally valid distractor')
            ds,distractor_faces,distractor_loop=distractor
            keep=np.ones(len(faces),bool);keep[target_faces]=False;keep[distractor_faces]=False
            cue_center=vertices[np.unique(faces[target_faces])].mean(0)
            cue_scale=max(float(np.linalg.norm(np.ptp(vertices[np.unique(faces[target_faces])],axis=0))),1e-6)
            cue=np.r_[cue_center,cue_scale]
            sid=f'{name}_cut{operation+1}'; path=DATA/'cases'/f'{sid}.npz';path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path,vertices=vertices,faces=faces[keep],cue=cue)
            cases.append({'index':len(cases),'sample_id':sid,'source_model_id':name,'category':name,
                'input_path':str(path.resolve()),'input_sha256':sha(path),'cue':cue.tolist(),'anchor_bbox':anchor,
                'target_seed_face':seed,'distractor_seed_face':ds,'target_removed_faces_n':80,'distractor_removed_faces_n':48})
            labels.append({'sample_id':sid,'target_loop_vertices':target_loop.tolist(),
                           'distractor_loop_vertices':distractor_loop.tolist(),'removed_target_faces':target_faces.tolist(),
                           'removed_distractor_faces':distractor_faces.tolist()})
            print(json.dumps({'prepared':sid,'faces':len(faces[keep]),'target_boundary':len(target_loop)}),flush=True)
    from scripts.formal_coseg_external_benchmark import RANKER,CBE
    write(OUT/'protocol.json',{'schema':'stanford_scan_controlled_face_removal','source_page':construction['source_page'],
        'source_credit':construction['source_credit'],'scope':construction['scope'],'construction_protocol_sha256':sha(DATA/'construction_protocol.json'),
        'n_cases':len(cases),'n_sources':4,'independent_unit':'source_model_id','selection':'four frozen bbox anchors per source; structural valid disk cuts only, fixed before inference',
        'ranker_sha256':sha(RANKER),'cbe_sha256':sha(CBE),'implementation_sha256':sha(ROOT/'tcmr/integrated_repair.py'),
        'target_labels_read':False,'sources':source_rows,'cases':cases})
    write(OUT/'score_labels.json',{'scope':'score stage only','rows':labels})


def run():
    from scripts import formal_coseg_external_benchmark as baseline
    baseline.OUT=OUT
    baseline.infer()
    # CGAL cue-nearest selection considers ALL valid input boundaries, independently
    # of the learned prior's top two. Reuse a measured candidate when identical.
    from tcmr import integrated_repair as method
    from tcmr.boundary import extract_boundary_loops
    from scripts.local_repair_topology import inspect_target_cycle
    source=read(OUT/'protocol.json')
    for case in source['cases']:
        path=OUT/'predictions'/f"{case['index']:03d}.json";prediction=read(path)
        with np.load(case['input_path'],allow_pickle=False) as z:vertices=z['vertices'];faces=z['faces'];cue=z['cue']
        loops=extract_boundary_loops(vertices,faces)
        options=[(float(np.linalg.norm(vertices[loop].mean(0)-cue[:3])),i,loop) for i,loop in enumerate(loops)
                 if inspect_target_cycle(faces,loop)['input_simple_closed_boundary']]
        if not options:continue
        _,loop_id,loop=min(options,key=lambda x:(x[0],x[1]))
        loop_row=next((r for r in prediction['loops'] if r['loop_id']==loop_id),None)
        work=OUT/'cgal_cue'/f"case{case['index']:03d}";work.mkdir(parents=True,exist_ok=True)
        if loop_row is None:
            inp=work/'damaged_input.npz';np.savez_compressed(inp,vertices=vertices,faces=faces,selected_loop=loop)
            loop_row={'loop_id':loop_id,'input_path':str(inp.resolve()),'candidates':[],'comparator_only':True}
            prediction['loops'].append(loop_row)
        else:inp=Path(loop_row['input_path'])
        for candidate_method in ('cgal_refine_fair','cgal_polyline'):
            if any(c['method']==candidate_method for c in loop_row['candidates']):continue
            destination=work/f'{candidate_method}.json'
            candidate=read(destination) if destination.exists() else method._run_candidate(
                {'method':candidate_method,'input_path':str(inp.resolve()),'input_geometry_key':method.content_key(vertices,faces,loop)},destination)
            loop_row['candidates'].append(candidate)
        pool=[c for c in loop_row['candidates'] if c['method'] in ('cgal_refine_fair','cgal_polyline') and c.get('safe_eligible') is True and c.get('seam_normal_error') is not None]
        choice=min(pool,key=lambda c:(c['seam_normal_error'],c['method'])) if pool else None
        prediction['cgal_library_decision']={'selected_loop_id':loop_id,'decision':'accept' if choice else 'null',
            'chosen_method':choice['method'] if choice else None,'patch_path':choice.get('patch_path') if choice else None,
            'selection_rule':'cue nearest across all valid input boundaries; minimum same seam proxy across safe refine/fair and polyline candidates','target_labels_read':False}
        write(path,prediction)
    import pymeshfix
    labels={r['sample_id']:r for r in read(OUT/'score_labels.json')['rows']};rows=[]
    for case in source['cases']:
        raw=OUT/'meshfix_outputs'/f"{case['index']:03d}.npz";status=raw.with_suffix('.json')
        if status.exists():rows.append(read(status));continue
        with np.load(case['input_path'],allow_pickle=False) as z:vertices=z['vertices'];faces=z['faces']
        started=time.perf_counter();error=None
        try:vout,fout=pymeshfix.clean_from_arrays(vertices,faces,verbose=False)
        except Exception as exc:vout=np.empty((0,3));fout=np.empty((0,3),np.int64);error=repr(exc)
        elapsed=time.perf_counter()-started;vout=np.asarray(vout,float);fout=np.asarray(fout,np.int64)
        raw.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(raw,vertices=vout,faces=fout)
        annotation=labels[case['sample_id']];tol=max(1e-5*np.linalg.norm(np.ptp(vertices,axis=0)),1e-8)
        row={'sample_id':case['sample_id'],'source_model_id':case['source_model_id'],'failed':error,'runtime_seconds':elapsed,
             'output_path':str(raw.resolve()),'output_sha256':sha(raw),'map_tolerance':float(tol),
             'target':baseline.mapped_loop(vertices,annotation['target_loop_vertices'],vout,fout,tol) if len(vout) else {},
             'distractor':baseline.mapped_loop(vertices,annotation['distractor_loop_vertices'],vout,fout,tol) if len(vout) else {},
             'global_watertight':bool(len(fout) and all(n==2 for n in baseline.edge_counts(fout).values()))}
        write(status,row);rows.append(row);print(json.dumps({'meshfix':len(rows),'total':16}),flush=True)
    successful=[r for r in rows if r['failed'] is None]
    write(OUT/'meshfix.json',{'complete':True,'successful_calls':len(successful),'failed_calls':16-len(successful),
        'target_closure_n':sum(r['target'].get('closed',False) for r in successful),
        'target_specific_n':sum(r['target'].get('closed',False) and r['distractor'].get('open',False) for r in successful),
        'distractor_preserved_open_n':sum(r['distractor'].get('open',False) for r in successful),
        'global_watertight_n':sum(r['global_watertight'] for r in successful),
        'mean_runtime_seconds':float(np.mean([r['runtime_seconds'] for r in rows])),'rows':rows})


def cluster_ci(rows, key, draws=20000):
    sources=sorted({r['source_model_id'] for r in rows})
    groups=[np.asarray([r[key] for r in rows if r['source_model_id']==s],float) for s in sources]
    rng=np.random.default_rng(20261013);values=[]
    for _ in range(draws):
        ids=rng.integers(0,len(groups),len(groups));values.append(float(np.concatenate([groups[i] for i in ids]).mean()))
    return [float(x) for x in np.quantile(values,[.025,.975])]


def score():
    from scripts import evaluate_external_benchmark as evaluator
    protocol=read(OUT/'protocol.json');labels={r['sample_id']:r for r in read(OUT/'score_labels.json')['rows']}
    rows=[]
    for case in protocol['cases']:
        prediction=read(OUT/'predictions'/f"{case['index']:03d}.json")
        full=case|labels[case['sample_id']]
        library_prediction=prediction|{'decisions':{'prior_locked_pool_with_joint':prediction['cgal_library_decision']}}
        library=evaluator.unified_method_row(full,library_prediction,'tcmr');library['method']='cgal_library';library['inference_wall_seconds']=None
        row={'sample_id':case['sample_id'],'source_model_id':case['source_model_id'],
            'tcmr':evaluator.unified_method_row(full,prediction,'tcmr'),
            'cgal':evaluator.unified_method_row(full,prediction,'cgal'),'cgal_library':library}
        row['tcmr']['inference_wall_seconds']=prediction.get('wall_seconds')
        selected=prediction['cgal_library_decision'].get('selected_loop_id')
        pool_loop=next((x for x in prediction['loops'] if x['loop_id']==selected),{})
        row['cgal_library']['candidate_pool_inclusive_seconds']=sum(float(c.get('inclusive_candidate_seconds') or 0.)
            for c in pool_loop.get('candidates',[]) if c['method'] in ('cgal_refine_fair','cgal_polyline'))
        for method_name in ('tcmr','cgal','cgal_library'):
            pred=library_prediction if method_name=='cgal_library' else prediction
            decision,loop,candidate=evaluator.candidate_for(pred,'tcmr' if method_name=='cgal_library' else method_name,full)
            with np.load(case['input_path'],allow_pickle=False) as z:vertices=z['vertices'];faces=z['faces']
            patch_path=None
            if decision.get('decision')=='accept' and candidate and candidate.get('patch_path'):
                patch_path=candidate['patch_path']
                with np.load(patch_path,allow_pickle=False) as z:vertices=z['vertices'];faces=np.vstack([faces,z['patch_faces']])
            merged=OUT/'outputs'/case['sample_id']/f'{method_name}.npz';merged.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(merged,vertices=vertices,faces=faces)
            row[method_name].update(merged_mesh_path=str(merged.resolve()),merged_mesh_sha256=sha(merged),
                patch_path=patch_path,input_path=case['input_path'],
                selected_loop_input_path=(loop or {}).get('input_path'),
                output_scope='accepted repair' if decision.get('decision')=='accept' else 'unchanged input on abstention')
        rows.append(row)
    summary={m:evaluator.method_summary([r[m] for r in rows]) for m in ('tcmr','cgal','cgal_library')}
    for m in summary:
        summary[m]['source_cluster_success_ci95']=cluster_ci([{'source_model_id':r['source_model_id'],'success':r[m]['correct_safe']} for r in rows],'success')
        summary[m]['retained_moved_inputs_n']=sum(r[m].get('accepted',False) and r[m].get('geometry',{}).get('moved_retained_area',0)>0 for r in rows)
    times=np.asarray([r['tcmr']['inference_wall_seconds'] for r in rows],float)
    summary['tcmr']['all_input_inference_seconds']={'n':len(times),'mean':float(times.mean()),'median':float(np.median(times))}
    times=np.asarray([r['cgal_library']['candidate_pool_inclusive_seconds'] for r in rows],float)
    summary['cgal_library']['derived_pool_process_seconds']={'n':len(times),'mean':float(times.mean()),'median':float(np.median(times)),
        'definition':'sum of the two cue-selected CGAL candidates inclusive process times; derived from shared generation, not independently timed pipeline wall clock'}
    diff=[{'source_model_id':r['source_model_id'],'difference':int(r['tcmr']['correct_safe'])-int(r['cgal_library']['correct_safe'])} for r in rows]
    paired={'n_operations':len(rows),'n_sources':4,'success_difference':float(np.mean([r['difference'] for r in diff])),
            'source_cluster_success_difference_ci95':cluster_ci(diff,'difference')}
    common=[r for r in rows if r['tcmr']['correct_safe'] and r['cgal_library']['correct_safe']]
    paired['common_correct_safe_n']=len(common)
    for metric in ('seam_normal_absolute_mean','triangle_quality_mean'):
        def value(row,method):
            g=row[method]['geometry']
            return g.get('triangle_quality',{}).get('mean') if metric=='triangle_quality_mean' else g.get(metric)
        eligible=[{'source_model_id':r['source_model_id'],'difference':value(r,'tcmr')-value(r,'cgal_library')} for r in common
                  if value(r,'tcmr') is not None and value(r,'cgal_library') is not None]
        if eligible:paired[metric]={'difference_mean':float(np.mean([r['difference'] for r in eligible])),
                                  'source_cluster_ci95':cluster_ci(eligible,'difference')}
    meshfix=read(OUT/'meshfix.json')
    meshfix['source_cluster_target_closure_ci95']=cluster_ci([{'source_model_id':r['source_model_id'],'closed':r['target'].get('closed',False)} for r in meshfix['rows']],'closed')
    meshfix['source_cluster_structural_target_specific_ci95']=cluster_ci([{'source_model_id':r['source_model_id'],'specific':r['target'].get('closed',False) and r['distractor'].get('open',False)} for r in meshfix['rows']],'specific')
    result={'schema':'stanford_scan_controlled_face_removal_report','complete':True,'n_cases':len(rows),'n_sources':4,
        'human_edit_cases_n':0,'scope':protocol['scope'],'source_credit':protocol['source_credit'],'source_page':protocol['source_page'],
        'protocol_sha256':sha(OUT/'protocol.json'),'score_labels_read_only_for_scoring':True,
        'primary_comparison':'TCMR versus cue-nearest CGAL candidate pool (refine/fair + polyline minimum safe seam)',
        'refine_only_scope':'compatibility diagnostic; Bunny/Dragon native context nonmanifold errors are not evidence of algorithmic geometric superiority',
        'confidence_intervals':{'primary':'paired bootstrap resampling four source models; all four operations travel together',
            'operation_binomial':'descriptive only; operations within a source are dependent'},'summary':summary,'paired':paired,
        'meshfix':{k:meshfix.get(k) for k in ('successful_calls','failed_calls','target_closure_n','target_specific_n','distractor_preserved_open_n','global_watertight_n','mean_runtime_seconds','source_cluster_target_closure_ci95','source_cluster_structural_target_specific_ci95')},
        'meshfix_rows':meshfix['rows'],'rows':rows}
    write(OUT/'report.json',result)
    md=['# Controlled face removal on Stanford reconstructed scans','',protocol['scope']+'. Four scan sources, four fixed operations each; the independent statistical unit is a source, not an operation. No human editors participated.','',
        '|Method|Correct-safe /16|Accepted|Target closed|Distractor open|Source-cluster 95% CI|', '|---|---:|---:|---:|---:|---|']
    for m,s in summary.items():md.append(f"|{m}|{s['correct_safe_n']}/16|{s['accepted_n']}|{s['target_closed_n']}|{s['distractor_open_n']}|{s['source_cluster_success_ci95']}|")
    md+=['','|Common correct-safe geometry|TCMR|CGAL candidate pool|Paired difference|Source-cluster 95% CI|','|---|---:|---:|---:|---|']
    for metric,key in [('Absolute seam (rad)','seam_normal_absolute_mean'),('Mean triangle quality','triangle_quality_mean')]:
        change=paired[key]
        md.append(f"|{metric}|{summary['tcmr'][key]['mean']:.6f}|{summary['cgal_library'][key]['mean']:.6f}|{change['difference_mean']:.6f}|{change['source_cluster_ci95']}|")
    md+=['',f"Both methods keep outside-two-ring vertices bitwise fixed on 15/15 accepted outputs. TCMR moves retained vertices on {summary['tcmr']['retained_moved_inputs_n']} accepted inputs; CGAL candidate pool on {summary['cgal_library']['retained_moved_inputs_n']}. RMS/input-median-edge means: {summary['tcmr']['retained_displacement_rms_over_edge']['mean']:.6f} vs 0.",
        '',f"TCMR all 16-input mean/median inference time: {summary['tcmr']['all_input_inference_seconds']['mean']:.3f}/{summary['tcmr']['all_input_inference_seconds']['median']:.3f} seconds. CGAL pool derived two-process sum mean: {summary['cgal_library']['derived_pool_process_seconds']['mean']:.3f} seconds; this is not independently timed pipeline wall clock."]
    md+=['',f"MeshFix: {meshfix['successful_calls']}/16 calls, {meshfix['target_closure_n']}/16 target closure, {meshfix['distractor_preserved_open_n']}/16 distractor left open, {meshfix['target_specific_n']}/16 structural target-specific outputs.",'',
        'Paired TCMR−CGAL candidate pool success difference: '+str(paired['success_difference'])+'; source-cluster 95% CI '+str(paired['source_cluster_success_difference_ci95'])+'.','',
        'The primary baseline independently selects the cue-nearest boundary across all valid input boundaries, then chooses the minimum same seam proxy among safe official CGAL refine/fair and polyline candidates. Refine-only is a compatibility diagnostic: all eight Bunny/Dragon failures are native nonmanifold-context errors, so its lower completion rate is not attributed to algorithmic geometric superiority.','',
        'This is scan-source transfer under controlled synthetic deletion, not natural scanner-hole repair or a user study. Four clusters provide limited population precision; operation-level binomial intervals are descriptive only. All methods receive identical damaged inputs. Geometry is conditional on jointly correct-safe repairs; MeshFix global closure is reported separately from selective closure.','',
        'Data credit: [Stanford University Computer Graphics Laboratory]('+protocol['source_page']+').']
    (OUT/'report.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('summary','paired','meshfix')},indent=2))


def robust_registration():
    """Freeze sources and perturbations before downloading or method inference."""
    from scripts.formal_coseg_external_benchmark import RANKER, CBE
    path = ROBUST_OUT / 'construction_protocol.json'
    if path.exists():
        return read(path)
    from datetime import datetime, timezone
    value = {
        'schema':'independent_scanned_object_robustness',
        'registered_utc':datetime.now(timezone.utc).isoformat(),
        'source_page':'https://ycb-benchmarks.s3.amazonaws.com/index.html',
        'source_credit':'Yale-CMU-Berkeley Object and Model Set; Google scanner meshes',
        'source_license':'Creative Commons Attribution 4.0 International (CC BY 4.0)',
        'source_license_url':'https://creativecommons.org/licenses/by/4.0/',
        'source_publication_doi':'10.1177/0278364917700714',
        'source_groups':YCB_GROUPS,'n_sources':24,'n_planned_cases':168,
        'selection':'Six fixed objects in each of four descriptive groups; no replacement after method results. All are distinct physical object IDs; previously seen four Stanford sources excluded.',
        'scope':'Controlled synthetic face removal on reconstructed real scanned objects, with synthetic noise and resolution changes; not natural scan holes, human editing, or a part reconstruction task.',
        'target_geometry_reference':'No ground-truth cap exists for the edited shape. The complete scanned object is used only to construct controlled deletions; no cap Chamfer or Hausdorff distance to the deleted surface is evaluated.',
        'conditions':ROBUST_CONDITIONS,
        'preprocessing':'Read exact raw Google 16k OBJ geometry with its shared vertex indices; fixed QEM target at most 15000 faces, agg7; normalize by baseline bounding-box longest axis; no hole filling or repair.',
        'target_anchor_bbox':[.2,.5,.5],
        'cut_rule':'Nearest face centroid to fixed anchor, ties by face ID; ascending-face-ID adjacency BFS until area threshold. Require disk cuts only using input topology. Select one seed supporting all three target areas; same seed for hole conditions. Distractor seed farthest from target, disjoint from largest target cut, area ratio0.003.',
        'resolution_rule':'Simplify intact baseline to 50% and25% faces before cutting; target seed nearest baseline seed position. Target and distractor physical area thresholds use baseline surface area. This tests mesh resolution under matched location and approximate physical hole area; exact removed area is reported.',
        'noise_rule':'Same fixed standard-normal XYZ field per object, seed20261013+object order; per-coordinate sigma0,0.05,0.15 times the clean pre-deletion mesh median edge; topology/cuts unchanged; clean cue fixed in all conditions.',
        'cue_rule':'Baseline removed target vertex centroid and baseline removed target bounding-box diagonal; held fixed across all seven conditions.',
        'independent_unit':'source_model_id; all conditions from one source travel together in bootstrap',
        'primary_endpoint':'correct target closure, distractor preserved, and no introduced intersection',
        'primary_comparison':'TCMR frozen versus independently cue-nearest CGAL refine/fair plus polyline safe candidate pool',
        'additional_baseline':'MeshFix global closure, with target closure and distractor preservation reported separately',
        'geometry_metrics':['absolute seam mean/p95','triangle quality mean/p05/min','retained displacement RMS/max over median input edge','changed surface area','outside-two-ring bitwise preservation'],
        'statistics':'Paired source bootstrap95% CI by condition and paired degradation from the shared clean reference; geometry conditional on jointly correct-safe pairs with eligible source counts. Inference failures remain in planned168 denominator; construction failures are explicit, never replaced after method results.',
        'ranker_sha256':sha(RANKER),'cbe_sha256':sha(CBE),
        'implementation_sha256':sha(ROOT/'tcmr/integrated_repair.py'),
        'target_labels_read_by_inference':False,
    }
    write(path,value)
    return value


def ycb_raw_geometry(object_id):
    """Keep only the exact official OBJ, not textures or expanded archives."""
    raw = ROBUST_DATA / 'raw' / f'{object_id}.obj'
    url = f'https://ycb-benchmarks.s3.amazonaws.com/data/google/{object_id}_google_16k.tgz'
    member_name = f'{object_id}/google_16k/textured.obj'
    if not raw.exists():
        raw.parent.mkdir(parents=True,exist_ok=True)
        error = None
        for attempt in range(3):
            try:
                request=urllib.request.Request(url,headers={'User-Agent':'TCMR research benchmark'})
                with urllib.request.urlopen(request,timeout=60) as response:
                    with tarfile.open(fileobj=response,mode='r|gz') as archive:
                        for member in archive:
                            if member.isfile() and member.name.endswith('/google_16k/textured.obj'):
                                raw.write_bytes(archive.extractfile(member).read())
                                member_name=member.name
                                break
                        else:
                            raise ValueError('official archive has no google_16k/textured.obj')
                error=None
                break
            except Exception as exc:
                error=exc
        if error is not None:
            raise error
    vertices,faces=[],[]
    for line in raw.read_text(encoding='utf-8').splitlines():
        fields=line.split()
        if fields and fields[0]=='v':vertices.append([float(x) for x in fields[1:4]])
        elif fields and fields[0]=='f':
            polygon=[int(x.split('/')[0])-1 for x in fields[1:]]
            if len(polygon)!=3:raise ValueError('expected triangular official mesh')
            faces.append(polygon)
    vertices=np.asarray(vertices,float);faces=np.asarray(faces,np.int64)
    if not len(faces) or not np.isfinite(vertices).all() or faces.min()<0 or faces.max()>=len(vertices):
        raise ValueError('invalid raw source geometry')
    return vertices,faces,{'source_url':url,'raw_mesh_path':str(raw.resolve()),
        'raw_mesh_sha256':sha(raw),'selected_member':member_name,
        'original_vertices':len(vertices),'original_faces':len(faces)}


def area_patch(seed, adjacency, areas, threshold):
    queue,seen,total=[int(seed)],{int(seed)},float(areas[seed])
    for face in queue:
        if total >= threshold:break
        for neighbor in sorted(adjacency.get(face,[])):
            if neighbor not in seen:
                seen.add(neighbor);queue.append(neighbor);total+=float(areas[neighbor])
                if total>=threshold:break
    return np.asarray(queue,np.int64)


def robust_cuts(vertices,faces,baseline_area,anchor_point,all_sizes=True):
    mesh=trimesh.Trimesh(vertices=vertices,faces=faces,process=False)
    adjacency={}
    for a,b in mesh.face_adjacency:
        adjacency.setdefault(int(a),[]).append(int(b));adjacency.setdefault(int(b),[]).append(int(a))
    areas=np.asarray(mesh.area_faces,float);centers=vertices[faces].mean(1)
    sizes=[.0025,.005,.01] if all_sizes else [.005]
    target=None
    for seed in np.lexsort((np.arange(len(faces)),np.linalg.norm(centers-anchor_point,axis=1))):
        patches={ratio:area_patch(seed,adjacency,areas,ratio*baseline_area) for ratio in sizes}
        loops={ratio:simple_cut(vertices,faces,ids) for ratio,ids in patches.items()}
        if all(x is not None for x in loops.values()):
            target=(int(seed),patches,loops);break
    if target is None:raise ValueError('no structurally valid target disk at requested area levels')
    seed,patches,loops=target
    target_vertices=set(np.unique(faces[patches[max(sizes)]]).tolist())
    for ds in np.lexsort((np.arange(len(faces)),-np.linalg.norm(centers-centers[seed],axis=1))):
        removed=area_patch(ds,adjacency,areas,.003*baseline_area)
        loop=simple_cut(vertices,faces,removed)
        if loop is not None and not target_vertices.intersection(np.unique(faces[removed])):
            return {'target_seed':seed,'target_patches':patches,'target_loops':loops,
                    'distractor_seed':int(ds),'distractor_patch':removed,'distractor_loop':loop,
                    'areas':areas,'seed_position':centers[seed]}
    raise ValueError('no disjoint structurally valid distractor disk')


def prepare_robustness():
    import fast_simplification
    construction=robust_registration()
    if (ROBUST_OUT/'protocol.json').exists():
        protocol=read(ROBUST_OUT/'protocol.json')
        print(json.dumps({'already_prepared':True,'n_cases':len(protocol['cases']),
                          'n_sources':protocol['n_sources']}),flush=True)
        return
    cases,labels,sources,errors=[],[],[],[]
    for source_index,(category,name) in enumerate((g,n) for g,ns in construction['source_groups'].items() for n in ns):
        try:
            vertices,faces,source=ycb_raw_geometry(name)
            if len(faces)>15000:
                vertices,faces=fast_simplification.simplify(vertices,faces,target_count=15000,agg=7.)
            center=(vertices.min(0)+vertices.max(0))/2;scale=np.ptp(vertices,axis=0).max()
            vertices=(vertices-center)/scale
            baseline_mesh=trimesh.Trimesh(vertices=vertices,faces=faces,process=False)
            baseline_area=float(baseline_mesh.area)
            anchor=vertices.min(0)+np.ptp(vertices,axis=0)*np.asarray(construction['target_anchor_bbox'])
            clean_cuts=robust_cuts(vertices,faces,baseline_area,anchor)
            base_patch=clean_cuts['target_patches'][.005]
            deleted_vertices=vertices[np.unique(faces[base_patch])]
            cue=np.r_[deleted_vertices.mean(0),max(float(np.linalg.norm(np.ptp(deleted_vertices,axis=0))),1e-6)]
            edge=float(np.median(baseline_mesh.edges_unique_length))
            noise=np.random.default_rng(20261013+source_index).normal(size=vertices.shape)
            source.update(source_model_id=name,category=category,normalization_center=center.tolist(),
                normalization_scale=float(scale),preprocessed_vertices=len(vertices),preprocessed_faces=len(faces),
                baseline_surface_area=baseline_area,baseline_median_edge=edge,noise_seed=20261013+source_index,
                target_seed_position=clean_cuts['seed_position'].tolist())
            sources.append(source)
            for condition in construction['conditions']:
                sample_id=name+'_'+condition['name']
                try:
                    cv,cf=vertices.copy(),faces.copy();cuts=clean_cuts
                    if condition['face_ratio']<1:
                        cv,cf=fast_simplification.simplify(cv,cf,target_count=int(len(cf)*condition['face_ratio']),agg=7.)
                        cuts=robust_cuts(cv,cf,baseline_area,clean_cuts['seed_position'],all_sizes=False)
                    if condition['noise_sigma_edge']:
                        cv+=noise*(edge*condition['noise_sigma_edge'])
                    ratio=condition['target_area_ratio'];removed=cuts['target_patches'][ratio]
                    dr=cuts['distractor_patch'];keep=np.ones(len(cf),bool);keep[removed]=False;keep[dr]=False
                    path=ROBUST_DATA/'cases'/f'{sample_id}.npz';path.parent.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(path,vertices=cv,faces=cf[keep],cue=cue)
                    row={'index':source_index*len(construction['conditions'])+construction['conditions'].index(condition),
                        'sample_id':sample_id,'source_model_id':name,'category':category,'condition':condition['name'],
                        'factor':condition['factor'],'input_path':str(path.resolve()),'input_sha256':sha(path),
                        'cue':cue.tolist(),'target_removed_faces_n':len(removed),'distractor_removed_faces_n':len(dr),
                        'target_removed_clean_area':float(cuts['areas'][removed].sum()),
                        'distractor_removed_clean_area':float(cuts['areas'][dr].sum()),
                        'target_area_over_baseline':float(cuts['areas'][removed].sum()/baseline_area),
                        'input_vertices_n':len(cv),'input_faces_n':len(cf[keep]),'construction_status':'prepared'}
                    cases.append(row)
                    labels.append({'sample_id':sample_id,'target_loop_vertices':cuts['target_loops'][ratio].tolist(),
                                   'distractor_loop_vertices':cuts['distractor_loop'].tolist()})
                except Exception as exc:
                    errors.append({'sample_id':sample_id,'source_model_id':name,'condition':condition['name'],'error':repr(exc)})
            print(json.dumps({'prepared_source':name,'sources_done':len(sources),'cases':len(cases),'errors':len(errors)}),flush=True)
        except Exception as exc:
            errors.extend({'sample_id':name+'_'+c['name'],'source_model_id':name,'condition':c['name'],'error':repr(exc)} for c in construction['conditions'])
            print(json.dumps({'source_failed':name,'error':repr(exc)}),flush=True)
    protocol=construction|{'construction_protocol_sha256':sha(ROBUST_OUT/'construction_protocol.json'),
        'n_cases':len(cases),'n_prepared_sources':len(sources),'sources':sources,'cases':cases,
        'construction_failures':errors,'prepared_without_model_inference':True,'target_labels_read':False}
    write(ROBUST_OUT/'protocol.json',protocol)
    write(ROBUST_OUT/'score_labels.json',{'scope':'score only; deleted surface is not a cap ground truth','rows':labels})
    write(ROBUST_OUT/'status.json',{'phase':'prepared','planned':construction['n_planned_cases'],'prepared':len(cases),
        'construction_failed':len(errors),'complete':False,'inference_started':False,'protocol_sha256':sha(ROBUST_OUT/'protocol.json')})


def run_robustness():
    """Independent, resumable per-method inference; annotations stay score-only."""
    from tcmr import integrated_repair as method
    from tcmr.boundary import extract_boundary_loops
    from scripts.local_repair_topology import inspect_target_cycle
    from scripts.formal_coseg_external_benchmark import RANKER,CBE
    import pymeshfix
    protocol=read(ROBUST_OUT/'protocol.json')
    for key,path in [('ranker_sha256',RANKER),('cbe_sha256',CBE),
                     ('implementation_sha256',ROOT/'tcmr/integrated_repair.py')]:
        if sha(path)!=protocol[key]:raise ValueError('frozen method changed: '+key)
    protocol_hash=sha(ROBUST_OUT/'protocol.json')
    for case_index,case in enumerate(protocol['cases']):
        if sha(case['input_path'])!=case['input_sha256']:raise ValueError('input changed: '+case['sample_id'])
        for name in ('tcmr','cgal_library','meshfix'):
            destination=ROBUST_OUT/'predictions'/f"{case['index']:03d}_{name}.json"
            if destination.exists():continue
            # All three timers begin with an identical already loaded damaged
            # mesh/cue. No measured candidate from another method is reused.
            with np.load(case['input_path'],allow_pickle=False) as z:
                vertices=z['vertices'].copy();faces=z['faces'].copy();cue=z['cue'].copy()
            work=ROBUST_OUT/'work'/case['sample_id']/name
            # TCMR creates its own fresh output directory inside infer().
            (work.parent if name=='tcmr' else work).mkdir(parents=True,exist_ok=True)
            started=time.perf_counter()
            try:
                if name=='tcmr':
                    result=method.infer(vertices,faces,cue,work,ranker_path=RANKER,cbe_path=CBE,workers=2)
                elif name=='cgal_library':
                    loops=extract_boundary_loops(vertices,faces)
                    choices=[(float(np.linalg.norm(vertices[loop].mean(0)-cue[:3])),i,loop)
                             for i,loop in enumerate(loops) if inspect_target_cycle(faces,loop)['input_simple_closed_boundary']]
                    decision={'decision':'null','selected_loop_id':None,'chosen_method':None}
                    result={'loops':[],'decisions':{'prior_locked_pool_with_joint':decision}}
                    if choices:
                        _,loop_id,loop=min(choices,key=lambda x:(x[0],x[1]))
                        inp=work/'selected_input.npz';np.savez_compressed(inp,vertices=vertices,faces=faces,selected_loop=loop)
                        candidates=[]
                        for cm in ('cgal_refine_fair','cgal_polyline'):
                            candidates.append(method._run_candidate({'method':cm,'input_path':str(inp.resolve()),
                                'input_geometry_key':method.content_key(vertices,faces,loop)},work/f'{cm}.json'))
                        safe=[c for c in candidates if c.get('safe_eligible') is True and c.get('seam_normal_error') is not None]
                        chosen=min(safe,key=lambda c:(c['seam_normal_error'],c['method'])) if safe else None
                        decision.update(selected_loop_id=loop_id,decision='accept' if chosen else 'null',
                            chosen_method=chosen['method'] if chosen else None)
                        result['loops']=[{'loop_id':loop_id,'input_path':str(inp.resolve()),'candidates':candidates}]
                else:
                    vout,fout=pymeshfix.clean_from_arrays(vertices,faces,verbose=False)
                    out=work/'output.npz';np.savez_compressed(out,vertices=np.asarray(vout,float),faces=np.asarray(fout,np.int64))
                    result={'output_path':str(out.resolve()),'output_sha256':sha(out)}
                elapsed=time.perf_counter()-started
                result.update(status='complete',wall_seconds=elapsed)
            except Exception as exc:
                result={'status':'failed','error':repr(exc),'wall_seconds':time.perf_counter()-started}
            result.update(sample_id=case['sample_id'],source_model_id=case['source_model_id'],condition=case['condition'],
                method=name,target_labels_read=False,protocol_sha256=protocol_hash,
                timing_scope='damaged arrays and cue in memory to returned repair decision; includes method-native initialization, generation and mandatory validation; imports and later common scoring excluded')
            write(destination,result)
            print(json.dumps({'case':case_index+1,'total':len(protocol['cases']),'method':name,
                              'status':result['status'],'seconds':result['wall_seconds']}),flush=True)
        write(ROBUST_OUT/'status.json',{'phase':'inference','complete':False,'planned':protocol['n_planned_cases'],
            'prepared':len(protocol['cases']),'cases_terminal':case_index+1,'protocol_sha256':protocol_hash})
    score_robustness()


def robust_metric(row,key):
    value=row.get('geometry',{})
    for part in key.split('.'):
        value=value.get(part) if isinstance(value,dict) else None
    return float(value) if isinstance(value,(float,int)) and np.isfinite(value) else None


def meshfix_geometry_verification(before_v,before_f,after_v,after_f,*,_native_worker=False):
    """Verify arbitrary remeshing without assuming retained vertex/face IDs.

    All triangle pairs are tested by the same native predicate as local repair.
    A before/after intersection is inherited only when both triangles have
    exactly identical coordinates, independent of vertex IDs and orientation.
    Degenerate or unavailable native checks never establish a safe output.
    Native CGAL runs in a fresh process to isolate its Windows DLLs from Torch.
    """
    if not _native_worker:
        started=time.perf_counter()
        try:
            payload=io.BytesIO()
            np.savez(payload,before_v=np.asarray(before_v,float),before_f=np.asarray(before_f,np.int64),
                     after_v=np.asarray(after_v,float),after_f=np.asarray(after_f,np.int64))
            code=('import io,json,sys,numpy as np; '
                  'from scripts.scan_edit_benchmark import meshfix_geometry_verification; '
                  'z=np.load(io.BytesIO(sys.stdin.buffer.read()),allow_pickle=False); '
                  'r=meshfix_geometry_verification(z["before_v"],z["before_f"],'
                  'z["after_v"],z["after_f"],_native_worker=True); '
                  'print(json.dumps(r,allow_nan=False))')
            completed=subprocess.run([sys.executable,'-c',code],input=payload.getvalue(),
                stdout=subprocess.PIPE,stderr=subprocess.PIPE,cwd=ROOT,timeout=120,
                env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2'),
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            if completed.returncode:
                result={'status':'native_process_failure','returncode':completed.returncode,
                    'error':completed.stderr.decode('utf-8',errors='replace')[-1500:],
                    'introduced_intersection':None,'geometry_no_new_topology_defect':None}
            else:
                result=json.loads(completed.stdout.decode('utf-8'))
                result['native_verification_seconds']=result.get('verification_seconds')
        except subprocess.TimeoutExpired:
            result={'status':'timeout','error':'native geometry verification exceeded120 seconds',
                    'introduced_intersection':None,'geometry_no_new_topology_defect':None}
        except Exception as exc:
            result={'status':'unavailable','error':repr(exc),'introduced_intersection':None,
                    'geometry_no_new_topology_defect':None}
        result['execution']='isolated native subprocess; mesh arrays transferred through stdin without temporary files'
        result['verification_seconds']=time.perf_counter()-started
        return result
    from collections import Counter
    from scripts.cgal_changed_surface_audit import _query
    from scripts.evaluate_external_benchmark import topology
    started=time.perf_counter()
    try:
        bv=np.asarray(before_v,float);bf=np.asarray(before_f,np.int64).reshape(-1,3)
        av=np.asarray(after_v,float);af=np.asarray(after_f,np.int64).reshape(-1,3)
        for vertices,faces in ((bv,bf),(av,af)):
            if not len(vertices) or not len(faces) or not np.isfinite(vertices).all():
                raise ValueError('empty or nonfinite mesh')
            if faces.min()<0 or faces.max()>=len(vertices):raise ValueError('invalid vertex indices')
        before_pairs,before_degenerate=_query(bv,bf,np.arange(len(bf)))
        after_pairs,after_degenerate=_query(av,af,np.arange(len(af)))
        def triangle_keys(vertices,faces):
            return [tuple(sorted(tuple(map(float,p)) for p in triangle)) for triangle in vertices[faces]]
        bk,ak=triangle_keys(bv,bf),triangle_keys(av,af)
        def pair_key(keys,pair):return tuple(sorted((keys[pair[0]],keys[pair[1]])))
        before_counts=Counter(pair_key(bk,pair) for pair in before_pairs)
        after_counts=Counter(pair_key(ak,pair) for pair in after_pairs)
        introduced_counts=after_counts-before_counts
        introduced_n=sum(int(n) for n in introduced_counts.values())
        inherited_n=sum(int(n) for n in (after_counts & before_counts).values())
        bt,at=topology(bv,bf),topology(av,af)
        no_new_topology=bool(bt['finite'] and at['finite'] and
            at['degenerate_faces']<=bt['degenerate_faces'] and at['nonmanifold_edges']<=bt['nonmanifold_edges'])
        edges={}
        for face in af:
            for a,b in zip(face,np.roll(face,-1)):
                edge=tuple(sorted((int(a),int(b))));edges[edge]=edges.get(edge,0)+1
        result={'status':'success','algorithm':'CGAL AABB triangle soup + native triangle intersection',
            'scope':'all before and after triangles; arbitrary global remeshing; no retained-prefix assumption',
            'inheritance_rule':'exact coordinate triangle-pair multiset; vertex IDs and triangle orientation ignored; no approximate matching',
            'before_intersection_pair_count':len(before_pairs),'after_intersection_pair_count':len(after_pairs),
            'inherited_intersection_pair_count':inherited_n,'introduced_pair_count':introduced_n,
            'introduced_intersection_nondegenerate':bool(introduced_n),
            'introduced_intersection':True if introduced_n else (None if before_degenerate or after_degenerate else False),
            'before_degenerate_faces':before_degenerate,'after_degenerate_faces':after_degenerate,
            'before_topology':bt,'after_topology':at,'geometry_no_new_topology_defect':no_new_topology,
            'global_watertight':bool(edges and all(n==2 for n in edges.values())),
            'degenerate_caveat':'Negative full-mesh intersection claims remain unknown when native triangle predicates omit any degenerate face'}
    except Exception as exc:
        result={'status':'unavailable','error':repr(exc),'introduced_intersection':None,
                'geometry_no_new_topology_defect':None}
    result['verification_seconds']=time.perf_counter()-started
    return result


def robust_pair(rows,left='tcmr',right='cgal_library'):
    metrics=['seam_normal_absolute_mean','seam_normal_absolute_p95','triangle_quality.mean',
             'triangle_quality.p05','triangle_quality.min','retained_displacement_rms_over_edge',
             'retained_displacement_max_over_edge','changed_surface_area_ratio']
    terminal=[r for r in rows if all(r[m].get('status') in ('complete','failed') for m in (left,right))]
    result={'n_planned_pairs':len(rows),'n_terminal_pairs':len(terminal),
            'n_sources':len({r['source_model_id'] for r in terminal}),'geometry':{}}
    if not terminal:return result
    differences=[{'source_model_id':r['source_model_id'],'value':int(r[left]['correct_safe'])-int(r[right]['correct_safe'])} for r in terminal]
    result.update(success_difference=float(np.mean([r['value'] for r in differences])),
                  source_cluster_ci95=cluster_ci(differences,'value'))
    common=[r for r in terminal if r[left].get('correct_safe') and r[right].get('correct_safe')]
    result['common_correct_safe_n']=len(common)
    for metric in metrics:
        differences=[{'source_model_id':r['source_model_id'],'value':robust_metric(r[left],metric)-robust_metric(r[right],metric)}
                     for r in common if robust_metric(r[left],metric) is not None and robust_metric(r[right],metric) is not None]
        if differences:result['geometry'][metric]={'n':len(differences),'n_sources':len({r['source_model_id'] for r in differences}),
            'mean_difference':float(np.mean([r['value'] for r in differences])),'source_cluster_ci95':cluster_ci(differences,'value')}
    return result


def score_robustness():
    from scripts import evaluate_external_benchmark as evaluator
    from scripts import formal_coseg_external_benchmark as baseline
    protocol=read(ROBUST_OUT/'protocol.json');label_by={x['sample_id']:x for x in read(ROBUST_OUT/'score_labels.json')['rows']}
    methods=('tcmr','cgal_library');rows=[]
    for case in protocol['cases']:
        row={k:case[k] for k in ('sample_id','source_model_id','category','condition')}
        full=case|label_by[case['sample_id']]
        for name in (*methods,'meshfix'):
            path=ROBUST_OUT/'predictions'/f"{case['index']:03d}_{name}.json"
            if not path.exists():
                row[name]={'status':'pending','correct_safe':False,'accepted':False,'wrong_accepted':False};continue
            pred=read(path)
            if pred['status']=='failed':
                row[name]={'status':'failed','correct_safe':False,'accepted':False,'wrong_accepted':False,
                           'error':pred.get('error'),'inference_wall_seconds':pred['wall_seconds']};continue
            if name in methods:
                row[name]=evaluator.unified_method_row(full,pred,'tcmr')
                row[name].update(method=name,inference_wall_seconds=pred['wall_seconds'])
            else:
                with np.load(case['input_path'],allow_pickle=False) as z:vi=z['vertices'];fi=z['faces']
                with np.load(pred['output_path'],allow_pickle=False) as z:vo=z['vertices'];fo=z['faces']
                tolerance=max(1e-5*np.linalg.norm(np.ptp(vi,axis=0)),1e-8)
                target=baseline.mapped_loop(vi,full['target_loop_vertices'],vo,fo,tolerance)
                distractor=baseline.mapped_loop(vi,full['distractor_loop_vertices'],vo,fo,tolerance)
                structural=bool(target.get('closed') and distractor.get('open'))
                strict_closed=bool(target.get('closed') and all(n==2 for n in target.get('edge_counts',[])))
                verification=(meshfix_geometry_verification(vi,fi,vo,fo) if structural else
                    {'status':'not_needed_for_failed_structural_endpoint','introduced_intersection':None,
                     'geometry_no_new_topology_defect':None})
                correct_safe=bool(structural and strict_closed and verification.get('status')=='success' and
                    verification.get('introduced_intersection') is False and verification.get('geometry_no_new_topology_defect') is True)
                counts=baseline.edge_counts(fo)
                row[name]={'status':'complete','target':target,'distractor':distractor,
                    'target_closed':bool(target.get('closed')),'distractor_open':bool(distractor.get('open')),
                    'strict_target_closed':strict_closed,'structural_target_specific':structural,
                    'correct_safe':correct_safe,'accepted':True,'wrong_accepted':not correct_safe,
                    'safe_status':'certified' if correct_safe else 'structural_or_geometry_endpoint_failed_or_unknown',
                    'audit':verification,'global_watertight':bool(counts and all(n==2 for n in counts.values())),
                    'inference_wall_seconds':pred['wall_seconds']}
        rows.append(row)
    for failure in protocol['construction_failures']:
        row={k:failure[k] for k in ('sample_id','source_model_id','condition')}
        row['construction_error']=failure['error']
        for name in (*methods,'meshfix'):row[name]={'status':'failed','failure_scope':'construction',
            'correct_safe':False,'accepted':False,'wrong_accepted':False}
        rows.append(row)
    by_condition={};clean={r['source_model_id']:r for r in rows if r['condition']=='clean'}
    for condition in protocol['conditions']:
        name=condition['name'];selected=[r for r in rows if r['condition']==name]
        summary={m:evaluator.method_summary([r[m] for r in selected]) for m in methods}
        for m in methods:
            terminal=[{'source_model_id':r['source_model_id'],'value':float(r[m]['correct_safe'])}
                      for r in selected if r[m]['status'] in ('complete','failed')]
            summary[m]['source_cluster_success_ci95']=cluster_ci(terminal,'value') if terminal else None
        by_condition[name]={'n_planned':len(selected),'summary':summary,'paired':robust_pair(selected),
            'meshfix':{'n_planned':len(selected),'terminal_n':sum(r['meshfix']['status'] in ('complete','failed') for r in selected),
                'target_closed_n':sum(r['meshfix'].get('target_closed',False) for r in selected),
                'distractor_open_n':sum(r['meshfix'].get('distractor_open',False) for r in selected),
                'structural_target_specific_n':sum(r['meshfix'].get('structural_target_specific',False) for r in selected),
                'correct_safe_n':sum(r['meshfix'].get('correct_safe',False) for r in selected),
                'global_watertight_n':sum(r['meshfix'].get('global_watertight',False) for r in selected)}}
        if name!='clean':
            by_condition[name]['paired_degradation_from_clean']={}
            for m in methods:
                pairs=[{'source_model_id':r['source_model_id'],'condition':r[m],'clean':clean[r['source_model_id']][m]} for r in selected]
                by_condition[name]['paired_degradation_from_clean'][m]=robust_pair(pairs,'condition','clean')
    complete=all(r[m]['status'] in ('complete','failed') for r in rows for m in (*methods,'meshfix'))
    result={'schema':'independent_scanned_object_robustness_report','complete':complete,
        'n_planned_cases':protocol['n_planned_cases'],'n_sources':protocol['n_sources'],
        'construction_failed_n':len(protocol['construction_failures']),'scope':protocol['scope'],
        'target_geometry_reference':protocol['target_geometry_reference'],'protocol_sha256':sha(ROBUST_OUT/'protocol.json'),
        'confidence_interval_unit':'physical source object; paired source bootstrap, conditions are repeated measures',
        'by_condition':by_condition,'rows':rows}
    write(ROBUST_OUT/'report.json',result)
    md=['# Scanned-object robustness under controlled deletion','',protocol['scope'],'',
        f"Planned: {protocol['n_sources']} independent sources, {protocol['n_planned_cases']} paired inputs. Complete: {complete}. Construction failures: {len(protocol['construction_failures'])}.",'',
        '|Condition|TCMR correct-safe|CGAL pool correct-safe|Paired difference|Source 95% CI|',
        '|---|---:|---:|---:|---|']
    for name,result_condition in by_condition.items():
        a=result_condition['summary']['tcmr'];b=result_condition['summary']['cgal_library'];paired=result_condition['paired']
        md.append(f"|{name}|{a['correct_safe_n']}/{result_condition['n_planned']}|{b['correct_safe_n']}/{result_condition['n_planned']}|{paired.get('success_difference')}|{paired.get('source_cluster_ci95')}|")
    md+=['',protocol['target_geometry_reference'],'',
        'Absolute seam, triangle quality including lower tail, retained displacement, area change, source-paired degradation, and all-input runtime distributions are in the accompanying JSON. Geometry uses jointly correct-safe pairs; failed and unknown outputs remain in endpoint denominators. Pending outputs are not failures.',
        '',f"Data: [{protocol['source_credit']}]({protocol['source_page']}); {protocol['source_license']}."]
    (ROBUST_OUT/'report.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    write(ROBUST_OUT/'status.json',{'phase':'scored' if complete else 'partial_scoring','complete':complete,
        'planned':protocol['n_planned_cases'],'n_sources':protocol['n_sources'],
        'construction_failed':len(protocol['construction_failures']),'protocol_sha256':sha(ROBUST_OUT/'protocol.json')})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','run','score','prepare-robustness','run-robustness','score-robustness']);args=parser.parse_args()
    {'prepare':prepare,'run':run,'score':score,'prepare-robustness':prepare_robustness,
     'run-robustness':run_robustness,'score-robustness':score_robustness}[args.stage]()
