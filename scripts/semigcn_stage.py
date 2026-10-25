"""Run a seeded official MGCN stage from a prepared input manifest."""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
ROOT=REPO/'experiments'/'semigcn_readiness_20261008_v1'
AUTHOR=REPO/'baselines'/'semigcn'


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def torch_environment():
    import torch
    return {"version": torch.__version__, "cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available()}


def main():
    global ROOT
    parser=argparse.ArgumentParser();parser.add_argument('--iterations',type=int,default=1)
    parser.add_argument('--stage',default='smoke1');parser.add_argument('--seed',type=int,default=314)
    parser.add_argument('--input-root',type=Path,help='Directory containing prepared_manifest.json')
    args=parser.parse_args()
    if args.input_root is not None: ROOT=args.input_root.resolve()
    if args.iterations<1: raise ValueError('Need positive iterations')
    if not 0 <= args.seed <= 2**32-1: raise ValueError('Seed must be an unsigned32-bit integer')
    manifest=json.loads((ROOT/'prepared_manifest.json').read_text()); source=Path(manifest['folder'])
    destination=ROOT/args.stage/source.name
    # Copy only declared inputs, never previous output manifests, caches or results.
    destination.mkdir(parents=True,exist_ok=False)
    for name,digest in manifest['files'].items():
        if sha(source/name)!=digest: raise RuntimeError('Prepared input hash changed')
        shutil.copy2(source/name,destination/name)
    sys.path.insert(0,str(AUTHOR));sys.path.insert(0,str(AUTHOR/'source'))
    import numpy as np
    import torch
    import pymeshlab as ml
    import sparse_cpu_compat
    from run_official import load_obj, seed_all
    seed_applications=[]
    def configured_seed(*_args, **_kwargs):
        # The author main calls its seed function without arguments before
        # model construction. Capture the requested experiment seed instead.
        seed_all(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        seed_applications.append({'requested_seed':args.seed,'torch_initial_seed':int(torch.initial_seed())})
    torch.set_num_threads(int(os.environ.get('SEMIGCN_THREADS', '2')))
    torch.set_num_interop_threads(1);configured_seed();sparse_cpu_compat.install()
    name=destination.name
    # Author smoothing operations, unchanged values (1 default + 30 uniform steps).
    ms=ml.MeshSet();ms.load_new_mesh(str(destination/f'{name}_initial.obj'))
    ms.apply_filter('laplacian_smooth',selected=True)
    ms.apply_filter('laplacian_smooth',stepsmoothnum=30,cotangentweight=False,selected=False)
    ms.save_current_mesh(str(destination/f'{name}_smooth.obj'))
    ini_v,ini_f=load_obj(destination/f'{name}_initial.obj')
    smooth_v,smooth_f=load_obj(destination/f'{name}_smooth.obj')
    if smooth_v.shape!=ini_v.shape or not np.array_equal(smooth_f,ini_f): raise RuntimeError('Smoothing altered topology')
    import mgcn_wo_gt as official
    official.torch_fix_seed=configured_seed;configured_seed()
    started=time.perf_counter()
    sys.argv=['mgcn_wo_gt.py','-i',destination.as_posix(),'-o',args.stage,'-iter',str(args.iterations)]
    official.main()
    outputs=list((destination/'output').glob(f'*_mgcn_{args.stage}/refine.obj'))
    if len(outputs)!=1: raise RuntimeError(f'Expected one fresh stage output, found {len(outputs)}')
    out_v,out_f=load_obj(outputs[0])
    if not np.array_equal(out_f,ini_f) or len(out_v)!=len(ini_v): raise RuntimeError('Model changed index correspondence')
    transform=json.loads((destination/'normalization_transform.json').read_text())
    physical=(out_v-np.asarray(transform['offset']))/transform['scale']
    if not np.isfinite(physical).all(): raise RuntimeError('Nonfinite model output')
    mapping=np.load(destination/'restore_mapping.npz'); merged=mapping['initializer_vertices'].copy()
    merged[mapping['global_ids']]=physical
    np.savez(destination/'restored_patch.npz',vertices=merged,patch_faces=mapping['patch_faces'])
    retained=len(mapping['vertices']); changed=np.linalg.norm(merged[:retained]-mapping['vertices'],axis=1)
    outside=np.ones(retained,bool); outside[mapping['global_ids'][mapping['global_ids']<retained]]=False
    mask_manifest=[]
    for p in sorted((destination/'dummy_mask').glob('*.pt')):
        t=torch.load(p);mask_manifest.append({'name':p.name,'shape':list(t.shape),'sha256':sha(p),
            'tensor_sha256':hashlib.sha256(t.cpu().numpy().tobytes()).hexdigest()})
    summary={'status':'completed','stage':args.stage,'iterations':args.iterations,'elapsed_seconds':time.perf_counter()-started,
      'seed':args.seed,'python_hash_seed':os.environ.get('PYTHONHASHSEED'),
      'torch_initial_seed':int(torch.initial_seed()),'seed_applications':seed_applications,
      'device':'cuda:0' if torch.cuda.is_available() else 'cpu','cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
      'seed_policy':'Requested seed controls Python/NumPy/Torch/CUDA before data and dummy-mask generation; the same requested seed resets the author model initializer',
      'source_commit':'fe17377db287298c77f325eeeeec5285b1c0f626','track':manifest['track'],
      'vertex_count':len(out_v),'face_count':len(out_f),'topology_preserved':True,'finite_output':True,
      'outside_component_bitwise_frozen':bool(np.array_equal(merged[:retained][outside],mapping['vertices'][outside])),
      'retained_displacement_mean':float(changed.mean()),'retained_displacement_max':float(changed.max()),
      'refine_obj':str(outputs[0]),'refine_sha256':sha(outputs[0]),'patch_sha256':sha(destination/'restored_patch.npz'),
      'dummy_masks':mask_manifest,'gt_geometry_read':False,'sealed_read':False,'use_for_paper_comparison':False,
      'interpretation':'Only input compatibility/full forward-backward-refinement functionality; no quality or convergence claim'}
    (destination/'stage_output_manifest.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
