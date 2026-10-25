"""OAT sensitivity of the frozen joint optimizer on a fixed validation prefix.

This measures sensitivity, and never replaces the frozen test configuration.
Each case shares a CGAL initializer; failed cases stay in every denominator.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import argparse, hashlib, json, os, subprocess, sys, time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'experiments/validation_final_sensitivity'
CONFIGS=[dict(name='default',steps=10,budget=.02,normal=10.),
         dict(name='steps5',steps=5,budget=.02,normal=10.),
         dict(name='steps20',steps=20,budget=.02,normal=10.),
         dict(name='budget001',steps=10,budget=.01,normal=10.),
         dict(name='budget004',steps=10,budget=.04,normal=10.),
         dict(name='normal5',steps=10,budget=.02,normal=5.),
         dict(name='normal20',steps=10,budget=.02,normal=20.)]
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')

def budget_input(case):
 """Recover the original ordered loop, without modifying the cached mesh."""
 from tcmr.boundary import extract_boundary_loops
 with np.load(ROOT/case['path'],allow_pickle=False) as z:
  vertices=np.asarray(z['vertices'],float);faces=np.asarray(z['faces'],np.int64)
  loops=extract_boundary_loops(vertices,faces)
  if 'target_loop' in z:
   target=set(map(int,z['target_loop']))
   matching=[L for L in loops if set(map(int,L))==target]
   if len(matching)!=1:raise ValueError('No exact extracted target boundary')
   loop=matching[0]
  else:loop=loops[int(case['target_loop_index'])]
 return vertices,faces,np.asarray(loop,np.int64)

def budget_verify(index,name):
 """An isolated native process: no mesh generation or optimizer calls."""
 from scripts.cgal_changed_surface_audit import audit_changed_surface
 case=json.loads((OUT/'protocol.json').read_text())['cases'][index]
 path=OUT/'cases'/f'{index:03d}'/(name+'_safety.json')
 try:
  before,faces,loop=budget_input(case)
  with np.load(path.with_name(name+'.npz'),allow_pickle=False) as z:
   vertices=np.asarray(z['vertices'],float);patch=np.asarray(z['patch_faces'],np.int64)
  value=audit_changed_surface(before,faces,vertices,patch,loop)
 except Exception as exc:value=dict(status='native_failure',introduced_intersection=None,error=repr(exc))
 dump(path,value)

def score_budgets():
 """Uniform native scoring of the three pre-existing displacement budgets."""
 from scripts.evaluate_external_benchmark import geometry_metrics
 protocol=json.loads((OUT/'protocol.json').read_text());cases=protocol['cases']
 names=['budget001','default','budget004'];start=time.perf_counter()
 previous_path=OUT/'budget_results.json'
 previous=json.loads(previous_path.read_text()) if previous_path.exists() else {}
 previous_rows={r['index']:r for r in previous.get('rows',[])} if previous.get('complete') else {}
 score_protocol=dict(schema='cached_budget_native_sensitivity',n=len(cases),budgets=[.01,.02,.04],
  source_protocol='experiments/validation_final_sensitivity/protocol.json',
  source_report='experiments/validation_final_sensitivity/report.json',
  population='All 30 preselected validation records, including the two cached initializer failures',
  generation='Cached meshes only; initializer, 10 steps, all weights, target and production defaults unchanged; no regeneration or tuning',
  eligibility='Native status success AND strict selected-loop closure true AND introduced_intersection false AND no new topology defect',
  intersection='Confirmed new nondegenerate intersections and unknown results (including any degenerate faces) are counted separately; unknown is rejected',
  statistics='20000 paired source-mesh bootstrap replicates, seed 20261025; all records in endpoint denominators; geometry on common valid records',
  normalization='Retained RMS uses median input edge h_in; optimizer bound uses original target-boundary median edge h',
  frozen_default_unchanged=True)
 score_protocol['configuration_provenance']=dict(
  recorded_per_arm=dict(steps=10,normal_weight=10.,budget=[.01,.02,.04]),
  current_runner_recipe='One CGAL refine-fair initializer per case; DEFAULT_WEIGHTS with normal=10; joint mode, default learning_rate=.02 and rings=2; only the budget varies',
  current_optimizer_sha256=hashlib.sha256((ROOT/'tcmr/joint_optimizer.py').read_bytes()).hexdigest(),
  historical_generation_optimizer_sha256=None,
  limitation='The historical cache records steps, normal weight, budget and output geometry, but not its imported optimizer source hash or a full weights/lr/rings execution snapshot. Its declared recipe agrees with the frozen defaults; byte-identical historical execution is not established.',
  interpretation='Cached development sensitivity at the recorded recipe, uniformly safety-scored now; no claim of a newly regenerated frozen-method experiment')
 dump(OUT/'budget_scoring_protocol.json',score_protocol)
 def one(index):
  case=cases[index];folder=OUT/'cases'/f'{index:03d}'
  original=json.loads((folder/'result.json').read_text());result=dict(index=index,**case,methods={})
  input_path=ROOT/case['path']
  result['input_sha256']=hashlib.sha256(input_path.read_bytes()).hexdigest()
  with np.load(input_path,allow_pickle=False) as z:
   meta=json.loads(str(z['meta_json'])) if 'meta_json' in z else {}
  result['source_model_id']=str(meta.get('model_id',case['model_id']))
  if str(case['model_id'])!=result['source_model_id']:raise ValueError('Source metadata mismatch')
  for name in names:
   cached=next(x for x in original['rows'] if x['name']==name)
   patch_path=folder/(name+'.npz');native_path=folder/(name+'_safety.json')
   if cached.get('status')!='success' or not patch_path.exists():
    result['methods'][name]=dict(valid=False,status=cached.get('status','missing_cached_mesh'),failure_class='construction_failure',generation=cached)
    continue
   t=time.perf_counter()
   mesh_sha=hashlib.sha256(patch_path.read_bytes()).hexdigest()
   saved_row=previous_rows.get(index,{})
   saved_method=saved_row.get('methods',{}).get(name,{})
   reuse=(saved_row.get('input_sha256')==result['input_sha256'] and saved_method.get('cached_mesh_sha256')==mesh_sha)
   verification=saved_method.get('verification') if reuse else None
   if verification is None and not native_path.exists():
    try:
     process=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--budget-verify',str(index),name],cwd=ROOT,
      env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'),
      capture_output=True,text=True,timeout=120,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
     if process.returncode or not native_path.exists():
      dump(native_path,dict(status='native_failure',introduced_intersection=None,returncode=process.returncode,error=process.stderr[-1500:]))
    except subprocess.TimeoutExpired:dump(native_path,dict(status='timeout',introduced_intersection=None))
   if verification is None:verification=json.loads(native_path.read_text())
   before,faces,loop=budget_input(case)
   with np.load(patch_path,allow_pickle=False) as z:
    vertices=np.asarray(z['vertices'],float);patch=np.asarray(z['patch_faces'],np.int64)
   geom=geometry_metrics(before,faces,vertices,patch,loop)
   valid=(verification.get('status')=='success' and verification.get('strict_selected_loop_closure') is True
    and verification.get('introduced_intersection') is False and geom.get('geometry_no_new_topology_defect') is True)
   failure=('valid' if valid else 'confirmed_intersection' if verification.get('introduced_intersection') is True else
    'native_unknown_or_failure' if verification.get('status')!='success' or verification.get('introduced_intersection') is None else
    'closure_failure' if verification.get('strict_selected_loop_closure') is not True else 'new_topology_defect')
   result['methods'][name]=dict(valid=bool(valid),status='scored',failure_class=failure,verification=verification,geometry=geom,
    cached_mesh=str(patch_path.relative_to(ROOT)),cached_mesh_sha256=mesh_sha,
    optimization_generation=cached,scoring_seconds=time.perf_counter()-t)
  return result
 rows=[]
 with ThreadPoolExecutor(max_workers=2) as ex:
  for fu in as_completed([ex.submit(one,i) for i in range(len(cases))]):
   row=fu.result();rows.append(row)
   print(json.dumps(dict(scored=len(rows),total=len(cases),index=row['index'],valid={k:v['valid'] for k,v in row['methods'].items()},elapsed=time.perf_counter()-start)),flush=True)
 rows.sort(key=lambda r:r['index'])
 source_counts=Counter(r['source_model_id'] for r in rows)
 if max(source_counts.values())!=1:raise ValueError('Repeated sources require source-cluster bootstrap')
 score_protocol['source_identity']=dict(distinct_model_ids=len(source_counts),record_n=len(rows),
  distinct_input_sha256=len({r['input_sha256'] for r in rows}),meta_json_model_ids_match_protocol=True,
  independent_unit='One PartNet model_id; 30 distinct source identities; input checksums are damaged-input identities, not original raw-source file hashes')
 metrics={'mean_seam_rad':lambda g:g['seam_normal_absolute_mean'],
  'p95_seam_rad':lambda g:g['seam_normal_absolute_p95'],
  'quality_mean':lambda g:g['triangle_quality']['mean'],'quality_p05':lambda g:g['triangle_quality']['p05'],
  'retained_rms_over_input_edge':lambda g:g['retained_displacement_rms_over_edge'],
  'changed_surface_area_ratio':lambda g:g['changed_surface_area_ratio']}
 def interval(values):
  values=np.asarray(values,float)
  if not len(values):return dict(n=0,mean=None,ci95=None)
  rng=np.random.default_rng(20261025);draws=rng.integers(0,len(values),(20000,len(values)))
  return dict(n=len(values),mean=float(values.mean()),ci95=np.quantile(values[draws].mean(1),[.025,.975]).tolist())
 summary={}
 for name in names:
  rs=[r['methods'][name] for r in rows];valid=[r for r in rs if r['valid']]
  summary[name]=dict(n=len(rs),budget=next(x['budget'] for x in CONFIGS if x['name']==name),valid_n=len(valid),
   failure_counts=dict(Counter(r['failure_class'] for r in rs)),
   generated_n=sum(r['status']=='scored' for r in rs),
   confirmed_intersection_n=sum(r.get('verification',{}).get('introduced_intersection') is True for r in rs),
   native_success_unknown_n=sum(r.get('verification',{}).get('status')=='success' and r['verification'].get('introduced_intersection') is None for r in rs),
   native_failure_or_timeout_n=sum(r.get('verification',{}).get('status') not in (None,'success') for r in rs),
   own_valid_geometry={k:interval([fn(r['geometry']) for r in valid]) for k,fn in metrics.items()})
 common=[r for r in rows if all(r['methods'][name]['valid'] for name in names)]
 common_summary={name:{k:interval([fn(r['methods'][name]['geometry']) for r in common]) for k,fn in metrics.items()} for name in names}
 comparisons={}
 for first,second in [('budget001','default'),('budget004','default'),('budget004','budget001')]:
  pair=[r for r in rows if r['methods'][first]['valid'] and r['methods'][second]['valid']]
  comparisons[first+'_minus_'+second]=dict(n_common_valid=len(pair),source_indices=[r['index'] for r in pair],
   validity_difference=interval([float(r['methods'][first]['valid'])-float(r['methods'][second]['valid']) for r in rows]),
   common_valid_geometry={k:interval([fn(r['methods'][first]['geometry'])-fn(r['methods'][second]['geometry']) for r in pair]) for k,fn in metrics.items()})
 output=dict(complete=True,protocol=score_protocol,summary=summary,n_common_valid_all_three=len(common),
  common_valid_all_three_indices=[r['index'] for r in common],common_valid_all_three_geometry=common_summary,
  paired=comparisons,rows=rows,wall_seconds=time.perf_counter()-start)
 dump(OUT/'budget_scoring_protocol.json',score_protocol);dump(OUT/'budget_results.json',output)
 print(json.dumps({k:v for k,v in output.items() if k not in ('rows','protocol')},indent=2),flush=True)
def worker(i):
 import torch
 torch.set_num_threads(1)
 from tcmr.boundary import extract_boundary_loops
 from tcmr.joint_optimizer import optimize_joint_collar,DEFAULT_WEIGHTS
 from tcmr.optimizer import triangle_quality
 from tcmr.validity import orient_patch_to_boundary
 from scripts.cgal_local_hole_adapter import repair
 from scripts.local_repair_topology import inspect_local_patch
 from scripts.evaluate_geometry_comparison import _seam_normal_error
 c=json.loads((OUT/'protocol.json').read_text())['cases'][i]
 dest=OUT/'cases'/f'{i:03d}';dest.mkdir(parents=True,exist_ok=True)
 start=time.perf_counter();rows=[]
 try:
  with np.load(ROOT/c['path'],allow_pickle=False) as z:
   v=np.asarray(z['vertices'],float);f=np.asarray(z['faces'],np.int64)
   loops=extract_boundary_loops(v,f)
   if 'target_loop' in z:
    target=set(map(int,z['target_loop']));ids=[j for j,L in enumerate(loops) if set(map(int,L))==target]
    if not ids:raise ValueError('No exact extracted target boundary')
    L=loops[ids[0]]
   else:L=loops[int(c['target_loop_index'])]
  init,patch,_=repair(v,f,L,'cgal_refine_fair');patch=orient_patch_to_boundary(patch,f,L)
  if not np.array_equal(init[:len(v)],v):raise ValueError('Initializer changed retained coordinates')
  np.savez_compressed(dest/'initializer.npz',vertices=init,patch_faces=patch,selected_loop=L)
 except Exception as exc:
  rows=[dict(**cfg,status='initializer_failure',safe=False,error=repr(exc)) for cfg in CONFIGS]
 else:
  for cfg in CONFIGS:
   t=time.perf_counter()
   try:
    weights=dict(DEFAULT_WEIGHTS);weights['normal']=cfg['normal']
    av,info=optimize_joint_collar(init,f,patch,L,len(v),steps=cfg['steps'],retained_displacement_budget=cfg['budget'],weights=weights)
    np.savez_compressed(dest/(cfg['name']+'.npz'),vertices=av,patch_faces=patch)
    topology=inspect_local_patch(f,L,patch)
    # Native intersection is deliberately omitted here: this is a validation
    # sensitivity study of the optimizer, while the frozen endpoint already
    # applies the complete native safety gate.
    rows.append(dict(**cfg,status='success',safe=None,seam=float(_seam_normal_error(av,f,patch,L)),quality=float(triangle_quality(av,patch).mean()),max_retained_displacement_h=float(np.linalg.norm(av[:len(v)]-v,axis=1).max()/info['boundary_median_edge']),outside_fixed=info['outside_support_bitwise_equal'],introduced_intersection='not_assessed',strict_closure=bool(topology['strict_target_closure']),seconds=time.perf_counter()-t))
   except Exception as exc:rows.append(dict(**cfg,status='variant_failure',safe=False,error=repr(exc),seconds=time.perf_counter()-t))
 result=dict(index=i,sample_id=c['sample_id'],split='val',rows=rows,wall_seconds=time.perf_counter()-start)
 dump(dest/'result.json',result)
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 if not (OUT/'protocol.json').exists():
  idx=json.loads((ROOT/'data/partnet_tcmr_90pct/dataset_index.json').read_text())
  from tcmr.boundary import extract_boundary_loops
  from scripts.cgal_local_hole_adapter import _validate_boundary
  cases=[]
  for r in idx['records']:
   if r['split']!='val': continue
   try:
    z=np.load(ROOT/r['path'],allow_pickle=False); loops=extract_boundary_loops(z['vertices'],z['faces'])
    L=np.asarray(z['target_loop'],np.int64) if 'target_loop' in z else loops[int(r['target_loop_index'])]
    _validate_boundary(z['vertices'],z['faces'],L)
   except Exception: continue
   cases.append(r)
   if len(cases)>=30: break
  dump(OUT/'protocol.json',dict(schema='final_joint_validation_oat',selection='First 30 validation records with a valid simple target boundary in locked historical index order; selection fixed before variant scoring',n=len(cases),configs=CONFIGS,cases=cases,frozen_default_unchanged=True,purpose='Sensitivity analysis of final backend, native intersection safety intentionally not assessed in this validation-only study'))
 cases=json.loads((OUT/'protocol.json').read_text())['cases'];start=time.perf_counter()
 def one(i):
  dest=OUT/'cases'/f'{i:03d}'/'result.json'
  if dest.exists():return json.loads(dest.read_text())
  env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
  try:
   p=subprocess.run([sys.executable,__file__,'--worker',str(i)],cwd=ROOT,env=env,capture_output=True,text=True,timeout=180,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
   if not dest.exists():raise RuntimeError(p.stderr[-1500:])
  except Exception as exc:dump(dest,dict(index=i,sample_id=cases[i]['sample_id'],rows=[dict(**cfg,status='case_failure',safe=False,error=repr(exc)) for cfg in CONFIGS]))
  return json.loads(dest.read_text())
 rows=[]
 with ThreadPoolExecutor(max_workers=2) as ex:
  for fu in as_completed([ex.submit(one,i) for i in range(len(cases))]):
   rows.append(fu.result());print(json.dumps(dict(done=len(rows),total=len(cases),elapsed=time.perf_counter()-start)),flush=True)
 rows.sort(key=lambda r:r['index']);summary={}
 for cfg in CONFIGS:
  rs=[next(x for x in r['rows'] if x['name']==cfg['name']) for r in rows];finite=[r for r in rs if r.get('seam') is not None];safe=[r for r in rs if r['safe']]
  summary[cfg['name']]=dict(n=len(rs),safe_n=len(safe),generation_success_n=len(finite),mean_seam_all_generated=float(np.mean([r['seam'] for r in finite])) if finite else None,mean_seam_safe=float(np.mean([r['seam'] for r in safe])) if safe else None,mean_quality_safe=float(np.mean([r['quality'] for r in safe])) if safe else None)
 default={r['sample_id']:next(x for x in r['rows'] if x['name']=='default') for r in rows}
 for cfg in CONFIGS:
  pairs=[(default[r['sample_id']],next(x for x in r['rows'] if x['name']==cfg['name'])) for r in rows];pairs=[(a,b) for a,b in pairs if a['safe'] and b['safe']]
  summary[cfg['name']]['paired_safe_n']=len(pairs)
  summary[cfg['name']]['paired_mean_seam_difference_vs_default']=float(np.mean([b['seam']-a['seam'] for a,b in pairs])) if pairs else None
 dump(OUT/'report.json',dict(complete=True,protocol=json.loads((OUT/'protocol.json').read_text()),summary=summary,rows=rows,wall_seconds=time.perf_counter()-start))
 print(json.dumps(summary,indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--worker',type=int)
 ap.add_argument('--score-budgets',action='store_true');ap.add_argument('--budget-verify',nargs=2);a=ap.parse_args()
 if a.budget_verify:budget_verify(int(a.budget_verify[0]),a.budget_verify[1])
 elif a.score_budgets:score_budgets()
 elif a.worker is not None:worker(a.worker)
 else:main()
