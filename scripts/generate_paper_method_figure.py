"""Render the method overview from a saved verified repair."""
from __future__ import annotations
import argparse,json,os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pyvista as pv

ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT
IMAGES=Path(os.environ.get('TCMR_PREVIEW_DIR', str(ROOT/'figures')))
FIGURE_DIR=Path(os.environ.get('TCMR_FIGURE_DIR', str(ROOT/'figures')))
OUTPUT=FIGURE_DIR/'method_overview.pdf'
CORAL,INK,LAVENDER,VIOLET,MAGENTA='#E38067','#271A92','#724BB6','#6F15AC','#A730A2'
METHODS=('cgal_refine_fair','cgal_polyline','center_fan','bounded_joint10')

def setup_font():
    folder=Path(os.environ.get('TCMR_HELVETICA_DIR', str(Path(os.environ.get('LOCALAPPDATA', str(Path.home())))/'HelveticaFonts')))
    for name in ('H-Helvetica-6__02__Helvetica.ttf','H-Helvetica-6__03__Helvetica-Bold.ttf'):
        font_manager.fontManager.addfont(str(folder/name))
    font_manager.findfont(font_manager.FontProperties(family='Helvetica'),fallback_to_default=False)
    plt.rcParams.update({'font.family':'Helvetica','font.size':12,'pdf.fonttype':42,
        'svg.fonttype':'none','figure.facecolor':'white','savefig.facecolor':'white',
        'text.color':'black','axes.unicode_minus':False,'mathtext.fontset':'custom',
        'mathtext.rm':'Helvetica','mathtext.it':'Helvetica:oblique','mathtext.bf':'Helvetica:bold'})

def tint(color,amount):
    rgb=np.array([int(color[i:i+2],16) for i in (1,3,5)])/255
    return tuple(1-amount+amount*rgb)

def unit(v):
    return np.asarray(v,dtype=float)/np.linalg.norm(v)

def poly(v,f):
    return pv.PolyData(v,np.column_stack([np.full(len(f),3),f]).ravel())

def load_case():
    directory=ROOT/'experiments'/'formal_coseg_external_benchmark'
    pred=json.loads((directory/'predictions'/'017.json').read_text(encoding='utf-8'))
    choice=pred['decisions']['prior_locked_pool_with_joint']
    assert choice['decision']=='accept' and choice['chosen_method']=='bounded_joint10'
    folder=directory/'work'/'case017'/f"loop{choice['locked_loop_id']}"
    def arrays(path):
        with np.load(path,allow_pickle=False) as z:
            return {key:z[key] for key in z.files}
    base=arrays(folder/'damaged_input.npz')
    candidates={key:arrays(folder/f'{key}.npz') for key in METHODS}
    assert np.array_equal(candidates[METHODS[0]]['patch_faces'],candidates[METHODS[-1]]['patch_faces'])
    protocol=json.loads((directory/'protocol.json').read_text(encoding='utf-8'))['cases'][17]
    other=next(row for row in pred['loops'] if row['loop_id']!=choice['locked_loop_id'])
    base['other_loop']=arrays(Path(other['input_path']))['selected_loop']
    base['cue']=np.asarray(protocol['cue'])[:3]
    v,loop=base['vertices'],base['selected_loop']
    center=v[loop].mean(0)
    _,_,vh=np.linalg.svd(v[loop]-center)
    normal=vh[-1]
    if np.dot(normal,center-v.mean(0))<0:normal*=-1
    base['local_center']=center
    base['local_radius']=np.linalg.norm(v[loop]-center,axis=1).max()
    base['local_direction']=unit(normal+.32*vh[0])
    base['local_up']=unit(np.cross(base['local_direction'],np.cross([0,0,1],base['local_direction'])))
    base['global_direction']=unit([-.28,.65,-1.0])
    base['global_up']=np.array([0.,1.,0.])
    edges=np.unique(np.sort(np.vstack([base['faces'][:,[0,1]],base['faces'][:,[1,2]],base['faces'][:,[2,0]]]),axis=1),axis=0)
    collar=np.unique(loop)
    for _ in range(2):
        collar=np.unique(np.r_[collar,edges[np.isin(edges,collar).any(1)].ravel()])
    base['collar']=collar
    base['h']=np.median(np.linalg.norm(v[loop]-v[np.roll(loop,-1)],axis=1))
    return base,candidates,pred

def rim(p,points,color,width=3):
    line=pv.lines_from_points(np.vstack([points,points[0]]))
    p.add_mesh(line,color=color,line_width=width,lighting=False,render_lines_as_tubes=True)

def render(base,candidate=None,*,local=False,support=False,cue=False,
           width=1400,height=1100,scale=1.0,highlight=True):
    """Render unmodified geometry with a shared orthographic camera.

    Local views crop in the viewport, leaving a natural surface context.
    No smoothing, displacement amplification, or per-method framing is applied.
    """
    v=base['vertices'] if candidate is None else candidate['vertices']
    f=base['faces']
    center,radius=base['local_center'],base['local_radius']
    if local:
        front=base['local_direction']
        up0=base['local_up']
        direction=unit(front+.24*up0)
        right=unit(np.cross(up0,direction));up=unit(np.cross(direction,right))
        focal=center
        parallel=radius*(2.42 if support else 1.86)*scale
        distance=radius*16
    else:
        direction=base['global_direction'];up0=base['global_up']
        right=unit(np.cross(up0,direction));up=unit(np.cross(direction,right))
        vv=np.vstack([base['vertices'],base['cue']])
        xx,yy=vv@right,vv@up
        focal=(base['vertices'].max(0)+base['vertices'].min(0))/2
        focal+=right*((xx.max()+xx.min())/2-focal@right)
        focal+=up*((yy.max()+yy.min())/2-focal@up)
        parallel=max(np.ptp(yy)/2,np.ptp(xx)/2*height/width)*1.10*scale
        distance=np.linalg.norm(np.ptp(vv,axis=0))*5
    p=pv.Plotter(off_screen=True,window_size=(width,height))
    p.set_background('white');p.enable_anti_aliasing('ssaa')
    # Matte, softly lit bodies retain volume without obscuring triangle edges.
    style=dict(show_edges=True,edge_color=tint(INK,.46 if local else .30),
        line_width=1.45 if local else .70,smooth_shading=True,
        ambient=.36,diffuse=.62,specular=.10,specular_power=24)
    if support:
        active=np.isin(f,base['collar']).any(1)
        p.add_mesh(poly(v,f[~active]),color=tint(LAVENDER,.10),**style)
        p.add_mesh(poly(v,f[active]),color=tint(LAVENDER,.37),**style)
        ids=base['collar']
        p.add_points(v[ids],color=tint(INK,.80),point_size=5,
                     render_points_as_spheres=True,lighting=False)
    else:
        p.add_mesh(poly(v,f),color=tint(LAVENDER,.13),**style)
    if candidate is not None:
        p.add_mesh(poly(v,candidate['patch_faces']),color=tint(CORAL,.87),
            show_edges=True,edge_color=tint(INK,.63),line_width=1.65 if local else 1.0,
            smooth_shading=False,ambient=.40,diffuse=.60,specular=.06)
    if highlight:
        rim(p,v[base['selected_loop']],CORAL if candidate is None else MAGENTA,
            3.0 if local else 3.6)
        if not local:rim(p,v[base['other_loop']],VIOLET,3.4)
    if cue:
        span=np.linalg.norm(np.ptp(base['vertices'],axis=0))
        p.add_mesh(pv.Sphere(radius=span*.013,center=base['cue']),color=CORAL,
                   ambient=.50,diffuse=.50,smooth_shading=True)
    p.camera.position=tuple(focal+direction*distance)
    p.camera.focal_point=tuple(focal);p.camera.up=tuple(up)
    p.camera.parallel_projection=True;p.camera.parallel_scale=parallel
    # Key, fill and soft frontal light share the same scene geometry.
    p.remove_all_lights()
    for offset,intensity in [(direction+.65*right+1.10*up,.78),
                             (direction-.90*right+.20*up,.36),
                             (direction,.32)]:
        p.add_light(pv.Light(position=tuple(focal+distance*offset),
            focal_point=tuple(focal),intensity=intensity,positional=False))
    p.reset_camera_clipping_range()
    im=p.screenshot(return_img=True)
    import vtk
    projected={}
    for name,point in [('target',v[base['selected_loop']].mean(0)),
                       ('other',v[base['other_loop']].mean(0)),('cue',base['cue'])]:
        coord=vtk.vtkCoordinate();coord.SetCoordinateSystemToWorld();coord.SetValue(*point)
        x,y=coord.GetComputedDoubleDisplayValue(p.renderer)
        projected[name]=(x/width,1-y/height)
    p.close()
    return im,projected


def canvas(w,h):
    fig=plt.figure(figsize=(w,h),dpi=180)
    ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,w),ylim=(0,h));ax.set_aspect('equal');ax.axis('off')
    ax.set_zorder(10);ax.patch.set_visible(False)
    return fig,ax

def label(ax,x,y,s,size=12,weight='normal',ha='center',color='black',**kw):
    return ax.text(x,y,s,fontsize=max(11.5,size),fontweight=weight,ha=ha,va='center',color=color,**kw)

def arrow(ax,a,b,color=INK,lw=1.2,style='-|>'):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle=style,mutation_scale=10,
        linewidth=lw,color=color,shrinkA=0,shrinkB=0))

def image_panel(fig,im,rect):
    w,h=fig.get_size_inches();x,y,rw,rh=rect
    ax=fig.add_axes([x/w,y/h,rw/w,rh/h]);ax.imshow(im,interpolation='lanczos');ax.axis('off')
    return ax

def save(fig,pdf,png):
    pdf.parent.mkdir(parents=True,exist_ok=True);png.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(pdf,metadata={'Title':'Target-conditioned mesh repair','Creator':'Python / Matplotlib and PyVista'})
    fig.savefig(png,dpi=450)
    print(pdf);print(png);plt.close(fig)

def draw(preview=None):
    setup_font();base,candidates,pred=load_case()
    w,h=7.14,4.22;fig,ax=canvas(w,h)
    for x,end,tag,title in [(0.12,2.10,'a','Spatial request'),
                            (2.34,4.91,'b','Candidate patches'),
                            (5.14,7.02,'c','Verified repair')]:
        label(ax,x,3.99,tag,size=13,weight='bold',ha='left',color=INK)
        label(ax,x+.23,3.99,title,size=12,weight='bold',ha='left')
        ax.plot([x,end],[3.76,3.76],color=tint(INK,.30),lw=.8)
    im,points=render(base,cue=True,width=1480,height=1340)
    ia=image_panel(fig,im,(.04,1.12,2.12,1.92))
    for key,txt,loc,col in [('other','Other\nopening',(.18,.98),VIOLET),
                            ('target','Requested rim',(.74,.92),CORAL),
                            ('cue','Cue',(.92,.77),CORAL)]:
        x,y=points[key]
        ia.annotate(txt,xy=(x,1-y),xytext=loc,xycoords='axes fraction',
            textcoords='axes fraction',fontsize=11.5,ha='center',va='center',
            color='black',arrowprops=dict(arrowstyle='-',color=col,lw=.85,
                                         shrinkA=4,shrinkB=5))
    label(ax,1.11,.75,'Boundary ranking',size=12,weight='bold')
    label(ax,1.11,.43,'Retain two proposals',size=11.5)
    arrow(ax,(2.10,2.20),(2.34,2.20),lw=1.35)
    label(ax,3.62,3.48,'Four patches per loop',size=11.5)
    positions=[(2.35,2.23),(3.69,2.23),(2.35,.79),(3.69,.79)]
    names=['Refine-fair','Polyline','Center fan','Joint refinement']
    for key,(x,y),name in zip(METHODS,positions,names):
        im,_=render(base,candidates[key],local=True,width=1120,height=980,scale=.88)
        image_panel(fig,im,(x,y,1.26,1.10))
        label(ax,x+.63,y-.19,name,size=11.5,
              weight='bold' if key==METHODS[-1] else 'normal')
        if key==METHODS[-1]:
            ax.plot([x+.13,x+1.13],[y-.37,y-.37],color=CORAL,lw=2.0)
    arrow(ax,(4.94,2.20),(5.16,2.20),lw=1.35)
    label(ax,6.08,3.48,'Lock target',size=12,weight='bold')
    label(ax,6.08,3.08,'Closure + intersection\nverification',size=11.5)
    im,points=render(base,candidates[METHODS[-1]],width=1480,height=1340)
    oa=image_panel(fig,im,(5.02,.98,2.12,1.92))
    x,y=points['target']
    oa.annotate('Patch',xy=(x,1-y),xytext=(.80,.92),xycoords='axes fraction',
        textcoords='axes fraction',fontsize=11.5,ha='center',color='black',
        arrowprops=dict(arrowstyle='-',color=CORAL,lw=.85,shrinkA=4,shrinkB=5))
    label(ax,6.08,.57,'Lowest seam among\nverified patches',size=11.5)
    save(fig,OUTPUT,preview or IMAGES/'method_framework_overview.png')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preview',type=Path)
    draw(parser.parse_args().preview)

