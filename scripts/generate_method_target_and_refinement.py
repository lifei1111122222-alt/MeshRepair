"""Show the saved patch/collar support and one fixed-plane seam section."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from matplotlib.patches import Circle
from generate_paper_method_figure import (
    setup_font, load_case, canvas, label, arrow, render, image_panel, save,
    PAPER, FIGURE_DIR, IMAGES, CORAL, INK, LAVENDER, VIOLET, MAGENTA, tint, METHODS, unit)
OUTPUT = FIGURE_DIR / 'method_target_and_refinement.pdf'

def seam_section(base, initial, final):
    """Select a readable edge from input only; slice both states in one plane."""
    loop=base['selected_loop'];rf=base['faces'];pf=initial['patch_faces']
    vi=initial['vertices'];view=unit(base['local_direction']+.24*base['local_up'])
    choices=[]
    for order,(a,b) in enumerate(zip(loop,np.roll(loop,-1))):
        rr=np.flatnonzero(np.isin(rf,[a,b]).sum(1)==2)
        pp=np.flatnonzero(np.isin(pf,[a,b]).sum(1)==2)
        if len(rr)!=1 or len(pp)!=1:continue
        ir,ip=int(rr[0]),int(pp[0]);normals=[];altitudes=[]
        edge=unit(vi[b]-vi[a]);mid=(vi[a]+vi[b])/2
        for face in (rf[ir],pf[ip]):
            points=vi[face]
            normals.append(unit(np.cross(points[1]-points[0],points[2]-points[0])))
            j=next(j for j in face if j not in (a,b));d=vi[j]-mid
            altitudes.append(np.linalg.norm(d-edge*(d@edge)))
        visible=min(n@view for n in normals)
        projected=np.linalg.norm(np.cross(edge,view))
        balance=min(altitudes)/max(altitudes)
        angle=float(np.arccos(np.clip(normals[0]@normals[1],-1,1)))
        if visible>=.25 and projected>=.75 and balance>=.30:
            choices.append(dict(a=int(a),b=int(b),ir=ir,ip=ip,order=order,
                angle=angle,visibility=float(visible),projection=float(projected),
                balance=float(balance)))
    if not choices:raise RuntimeError('No readable edge satisfies the input-only rule.')
    median=float(np.median([r['angle'] for r in choices]))
    selected=min(choices,key=lambda r:(abs(r['angle']-median),r['order']))
    a,b,ir,ip=(selected[k] for k in ('a','b','ir','ip'))
    origin=(vi[a]+vi[b])/2;tangent=unit(vi[b]-vi[a])
    jr=next(j for j in rf[ir] if j not in (a,b))
    jp=next(j for j in pf[ip] if j not in (a,b))
    direction=vi[jr]-origin;direction-=tangent*(direction@tangent)
    xx=-unit(direction);yy=unit(np.cross(tangent,xx))
    if (vi[jp]-origin)@yy>0:yy*=-1
    def intersect(triangle):
        distances=(triangle-origin)@tangent;points=[]
        for i,j in ((0,1),(1,2),(2,0)):
            di,dj=distances[i],distances[j]
            if abs(di)<1e-12:points.append(triangle[i])
            if di*dj<0:
                points.append(triangle[i]+di/(di-dj)*(triangle[j]-triangle[i]))
        unique=[]
        for point in points:
            if not any(np.linalg.norm(point-prior)<1e-10 for prior in unique):
                unique.append(point)
        if len(unique)!=2:raise RuntimeError('The fixed plane must cut each face twice.')
        local=np.array([[((p-origin)@xx)/base['h'],((p-origin)@yy)/base['h']]
                        for p in unique])
        return local[np.argsort(local[:,0])]
    states=[];angles=[]
    for data in (initial,final):
        v=data['vertices']
        states.append([intersect(v[rf[ir]]),intersect(v[pf[ip]])]);normals=[]
        for face in (rf[ir],pf[ip]):
            points=v[face]
            normals.append(unit(np.cross(points[1]-points[0],points[2]-points[0])))
        angles.append(float(np.arccos(np.clip(normals[0]@normals[1],-1,1))))
    metadata=dict(selected,readable_edges=len(choices),initial_median_rad=median,
        section_origin=origin.tolist(),section_normal=tangent.tolist(),
        initial_dihedral_deg=float(np.degrees(angles[0])),
        refined_dihedral_deg=float(np.degrees(angles[1])),
        initial_dihedral_rad=angles[0],refined_dihedral_rad=angles[1],
        selection_rule='initial-only: front-facing face normals >=0.25; projected edge >=0.75; adjacent-face altitude ratio >=0.30; nearest median initial oriented dihedral, then loop order',
        selection_uses_final_geometry=False,section_plane_fixed=True,
        displacement_exaggeration=1)
    return states,angles,origin,metadata

def project_local(base,point,*,width,height,scale):
    """Match the shared support renderer's orthographic camera."""
    direction=unit(base['local_direction']+.24*base['local_up'])
    right=unit(np.cross(base['local_up'],direction))
    up=unit(np.cross(direction,right))
    parallel=base['local_radius']*2.42*scale
    d=point-base['local_center']
    return np.array([.5+(d@right)/(2*parallel*width/height),
                     .5+(d@up)/(2*parallel)])

def draw(preview=None):
    setup_font();base,candidates,_=load_case()
    initial,final=candidates[METHODS[0]],candidates[METHODS[-1]]
    states,angles,seam_midpoint,metadata=seam_section(base,initial,final)
    fig,ax=canvas(7.14,4.36)
    label(ax,.12,4.12,'a',size=12,weight='bold',ha='left')
    label(ax,.36,4.12,'Patch and retained collar',size=12,weight='bold',ha='left')
    rw,rh,zoom=1570,1240,.82
    im,_=render(base,initial,local=True,support=True,width=rw,height=rh,scale=zoom)
    ia=image_panel(fig,im,(.08,.88,3.72,2.94))
    loc=project_local(base,seam_midpoint,width=rw,height=rh,scale=zoom)
    ia.scatter(*loc,s=31,facecolors=MAGENTA,edgecolors='white',linewidths=.9,
               transform=ia.transAxes,zorder=5)
    ax.scatter([.28],[.25],s=27,c=MAGENTA,zorder=5)
    label(ax,.41,.25,'Section edge',size=11.5,ha='left')
    for x,color,text in ((.28,tint(CORAL,.87),'Patch'),
                         (1.38,tint(LAVENDER,.37),'Collar'),
                         (2.59,tint(LAVENDER,.10),'Fixed')):
        ax.scatter([x],[.65],s=40,c=[color],marker='s',edgecolors=[INK],linewidths=.5)
        label(ax,x+.13,.65,text,size=12,ha='left')
    label(ax,3.98,4.12,'b',size=12,weight='bold',ha='left')
    label(ax,4.22,4.12,'Seam cross-section',size=12,weight='bold',ha='left')
    all_points=np.concatenate([seg for state in states for seg in state])
    section_scale=.57/max(abs(all_points[:,0]).max(),.2)
    for cx,state,angle,text in zip((4.65,6.28),states,angles,('Initial','Refined')):
        label(ax,cx,3.64,text,size=12);cy=2.90
        for segment,color in zip(state,(LAVENDER,CORAL)):
            points=segment*section_scale+[cx,cy]
            ax.plot(points[:,0],points[:,1],color=color,lw=3,
                    solid_capstyle='round',zorder=4)
            normal=unit([-(points[1,1]-points[0,1]),points[1,0]-points[0,0]])
            if normal[1]<0:normal*=-1
            middle=points.mean(0)
            arrow(ax,tuple(middle),tuple(middle+.40*normal),color=color,lw=1.3)
        junction=(state[0][np.argmax(state[0][:,0])]+
                  state[1][np.argmin(state[1][:,0])])/2
        junction=junction*section_scale+[cx,cy]
        ax.scatter(*junction,s=16,c=MAGENTA,zorder=6)
        label(ax,cx,2.38,rf'$\theta_e={angle:.3f}$ rad',size=12)
    label(ax,5.46,2.02,'Same edge and section plane',size=11.5)
    label(ax,3.98,1.49,'c',size=12,weight='bold',ha='left')
    label(ax,4.22,1.49,'Bounded retained motion',size=12,weight='bold',ha='left')
    center,radius=np.array([4.67,.72]),.43
    direction=unit([1.,.62])
    boundary,trial=center+radius*direction,center+.79*direction
    ax.add_patch(Circle(center,radius,facecolor=tint(LAVENDER,.10),
                        edgecolor=LAVENDER,lw=1.15))
    ax.plot([center[0],trial[0]],[center[1],trial[1]],color=LAVENDER,
            lw=.95,ls=(0,(2,2)))
    ax.scatter(*center,s=15,c=INK,zorder=5)
    ax.scatter(*trial,s=23,facecolors='white',edgecolors=CORAL,linewidths=1.1,zorder=5)
    ax.scatter(*boundary,s=24,c=VIOLET,zorder=6)
    arrow(ax,tuple(trial),tuple(boundary),color=VIOLET,lw=1.2)
    label(ax,center[0]-.14,center[1]-.18,r'$\mathbf{x}^{0}$',size=12)
    label(ax,6.12,.91,'Projection',size=12)
    label(ax,6.12,.57,r'$\|\Delta\mathbf{x}\|\leq0.02h$',size=12)
    label(ax,5.46,.17,'Exterior and connectivity fixed',size=11.5)
    metadata['output_pdf']=str(OUTPUT)
    print('SEAM_SECTION '+json.dumps(metadata,ensure_ascii=True,sort_keys=True))
    save(fig,OUTPUT,preview or IMAGES/'method_framework_detail.png')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preview',type=Path)
    draw(parser.parse_args().preview)
