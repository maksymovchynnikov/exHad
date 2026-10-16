"""Inelastic dark matter: chi2 -> chi1 + an exclusive electromagnetic final state.

The current's exclusive rates and primary Dalitz functions are exHad inputs.
The dark tensor is spin averaged; a pure parity-conserving vector production
leaves an unpolarized individual chi2. Correlations with the unobserved chi1
spin need not be propagated. Full visible-current angular correlations remain.
Four-vectors use (px,py,pz,E); widths are in GeV at epsilon^2=1.
Here m1 = mediator/ratio and m2 = (1+delta)*m1. The energy release must
be above the electron-pair threshold and below 1.7 GeV, where exHad's
explicit electromagnetic channels end. The mediator must be off shell.
Four-or-more-body channels retain the study's phase-space approximation;
this is not a general polarized-current fragmentation model.
"""
from functools import lru_cache
import json
import numpy as np
from scipy.integrate import cumulative_trapezoid
from . import ROOT, DATA, exclusive
from .full_decays import primary
from .secondary import decayer
from .core.pdg import m0
ALPHA=1/137.035999084
HBARC=1.973269804e-16
ME=.00051099895
MAX_CURRENT_MASS=1.7  # Strict upper bound of exclusive.rates('dark-photon', q).

def _parameters(mediator,ratio,delta,alpha_D):
    if not all(np.isfinite(x) and x>0 for x in (mediator,ratio,delta,alpha_D)):
        raise ValueError('mediator, ratio, delta and alpha_D must be finite and positive')
    m1=mediator/ratio;m2=m1*(1+delta);end=m2-m1
    if not 2*ME<end<MAX_CURRENT_MASS:
        raise ValueError('chi2 - chi1 mass splitting must be above 2 m_e and below 1.7 GeV')
    if mediator<=end:
        raise ValueError('This off-shell decay requires mediator mass > chi2 - chi1')
    return m1,m2,end

@lru_cache(maxsize=1)
def catalog():
    return exclusive.rows('dark-photon')

@lru_cache(maxsize=1)
def table_data():
    data=json.loads((DATA/'dark-photon/eventcalc_branching_ratios.json').read_text())
    return {r[0]:r for r in data}

def lepton_width(q,ml):
    q=np.asarray(q);r=ml*ml/q**2
    return ALPHA*q/3*np.sqrt(np.maximum(1-4*r,0))*(1+2*r)

def current_widths(q):
    """Absolute spectral widths fixed by the analytic electron width and exHad ratios.

    This avoids interpolation of the lifetime across the electron threshold.
    Leptonic widths are analytic; hadronic exclusive ratios use exHad's tables.
    No interpolation across a closed physical threshold is permitted.
    """
    q=np.atleast_1d(q).astype(float)
    if np.any(~np.isfinite(q)) or np.any(q<=2*ME) or np.any(q>=MAX_CURRENT_MASS):raise ValueError('Current mass must satisfy 2 m_e < q < 1.7 GeV')
    rows=catalog();ge=lepton_width(q,ME)
    widths=np.zeros((len(q),len(rows)))
    physical=q>=exclusive.table_start('dark-photon')
    br=np.zeros_like(widths)
    for j in np.flatnonzero(physical):br[j]=exclusive.rates('dark-photon',float(q[j]))
    be=br[:,next(i for i,r in enumerate(rows) if r.label=='e-pair')]
    for i,row in enumerate(rows):
        if row.label=='e-pair':widths[:,i]=ge
        elif row.label=='mu-pair':widths[:,i]=lepton_width(q,.1056583755)
        elif row.label=='tau-pair':continue
        elif row.has_hadrons:
            widths[:,i]=ge*np.divide(br[:,i],be,out=np.zeros_like(ge),where=be>0)
    return widths

def differential_width(s,m1,m2,mediator,alpha_D=.1,widths=None):
    """dGamma/dq^2 per epsilon^2, including the exact vector transition tensor.

    H.P = ((m2-m1)^2-s)*((m2+m1)^2+2s)/s.
    The massless/small-splitting integral tends to
    4 alpha alpha_D (m2-m1)^5/(15 pi mediator^4).
    """
    s=np.atleast_1d(s);q=np.sqrt(s)
    g=current_widths(q) if widths is None else np.asarray(widths)
    if g.ndim==1:g=g[:,None]
    d=(m2-m1)**2;ss=(m2+m1)**2
    k=alpha_D/(4*np.pi*m2**3)*np.maximum(d-s,0)*(ss+2*s)*np.sqrt(np.maximum((d-s)*(ss-s),0))/(q*(mediator*mediator-s)**2)
    return k[:,None]*g

@lru_cache(maxsize=64)
def spectrum(mediator,ratio=3.,delta=.1,alpha_D=.1,nodes=3073):
    m1,m2,end=_parameters(mediator,ratio,delta,alpha_D)
    if isinstance(nodes,bool) or not isinstance(nodes,(int,np.integer)) or nodes<3:
        raise ValueError('nodes must be an integer >= 3')
    # Uniform sqrt(q-2me) nodes resolve threshold and hadronic openings.
    q=2*ME+(end-2*ME)*np.linspace(1e-7,1,nodes)**2
    # Resolve the first open node even within a few keV of the e+e- threshold.
    q=np.maximum(q,np.nextafter(2*ME,np.inf))
    g=differential_width(q*q,m1,m2,mediator,alpha_D)*2*q[:,None]
    c=np.vstack([np.zeros(g.shape[1]),cumulative_trapezoid(g,q,axis=0)])
    return q,c,g

def width(mediator,**kwargs):return float(spectrum(float(mediator),**kwargs)[1][-1].sum())
def ctau(mediator,epsilon2=1.,**kwargs):
    """Proper decay length in metres for the supplied squared kinetic mixing."""
    if not np.isfinite(epsilon2) or epsilon2<=0:raise ValueError('epsilon2 must be finite and positive')
    return HBARC/(epsilon2*width(mediator,**kwargs))

def directions(rng,n):
    c=rng.uniform(-1,1,n);p=rng.uniform(0,2*np.pi,n);s=np.sqrt(1-c*c)
    return np.column_stack([s*np.cos(p),s*np.sin(p),c])

def boost(p4,frame):
    """Broadcastable boost from the frame's rest system, including its energy."""
    p4=np.asarray(p4,float);frame=np.asarray(frame,float)
    m=np.sqrt(np.maximum(frame[...,3]**2-(frame[...,:3]**2).sum(axis=-1),1e-30))
    b=frame[...,:3]/frame[...,3,None];g=frame[...,3]/m;b2=(b*b).sum(axis=-1)
    dot=(p4[...,:3]*b).sum(axis=-1)
    factor=np.divide((g-1)*dot,b2,out=np.zeros_like(dot),where=b2>0)+g*p4[...,3]
    return np.concatenate([p4[...,:3]+factor[...,None]*b,(g*(p4[...,3]+dot))[...,None]],axis=-1)

def rotate_z(v,axis):
    axis=np.asarray(axis);ref=np.zeros_like(axis);ref[:,1]=1;ref[abs(axis[:,1])>.9]=[1,0,0]
    x=np.cross(ref,axis);x/=np.linalg.norm(x,axis=1)[:,None];y=np.cross(axis,x)
    return v[:,0,None]*x+v[:,1,None]*y+v[:,2,None]*axis

def angular_weight(kind,c,q,m1,m2,ml=0.):
    # The common (Delta^2-q^2) cancels in the normalized tensor.
    ht=np.ones_like(np.asarray(q));hl=(m1+m2)**2/np.asarray(q)**2;tr=2*ht+hl
    hnn=ht+(hl-ht)*c*c
    if kind in ('pipi','threepi'):return 3*hnn/tr
    beta2=1-4*ml*ml/np.asarray(q)**2 if kind=='lepton' else 1.
    return (tr-beta2*hnn)/(tr*(1-beta2/3))

def generate(mediator,count,*,seed=1,ratio=3.,delta=.1,alpha_D=.1):
    """Complete chi2 rest decays; array of (px,py,pz,E,m,PDG), chi1 included."""
    if isinstance(count,bool) or not isinstance(count,(int,np.integer)) or count<1:
        raise ValueError('count must be a positive integer')
    if isinstance(seed,bool) or not isinstance(seed,(int,np.integer)) or seed<0:
        raise ValueError('seed must be a nonnegative integer')
    m1,m2,_=_parameters(mediator,ratio,delta,alpha_D)
    qgrid,cdf,_=spectrum(float(mediator),ratio,delta,alpha_D)
    rng=np.random.default_rng(seed);rates=cdf[-1];labels=rng.choice(len(rates),count,p=rates/rates.sum())
    qs=np.empty(count)
    for i in np.unique(labels):
        mask=labels==i;qs[mask]=np.interp(rng.uniform(0,rates[i],mask.sum()),cdf[:,i],qgrid)
    primary_rows=[];hadronic4pi=0
    # The chosen virtual photon points along +z before a common isotropic rotation.
    for q,i in zip(qs,labels):
        row=catalog()[i];ids=row.pdgs
        if len(ids)==2:
            masses=np.array([m0(p) for p in ids]);en=(q*q+masses[0]**2-masses[1]**2)/(2*q);mom=np.sqrt(max(0,en*en-masses[0]**2))
            kind='lepton' if abs(ids[0]) in [11,13] else 'pipi' if set(ids) in ({211,-211},{321,-321},{130,310}) else 'radiative'
            while True:
                n=directions(rng,1)[0];w=angular_weight(kind,n[2],q,m1,m2,masses[0])
                if rng.uniform(0,3)<=w:break
            a=np.array([[*(mom*n),en,masses[0],ids[0]],[*(-mom*n),q-en,masses[1],ids[1]]])
        else:
            # exHad owns the internal Dalitz distribution; rotate its current
            # relative to the dark transition only after this generation.
            expression=table_data()[row.label][3]
            while True:
                a=np.asarray(primary(float(q),list(ids),1,int(rng.integers(1,800000000)),expression)).reshape(-1,8)[:,:6]
                if len(ids)==3 and set(ids)=={211,-211,111}:
                    normal=np.cross(a[0,:3],a[1,:3]);cc=normal[2]/np.linalg.norm(normal)
                    if rng.uniform(0,3)>angular_weight('threepi',cc,q,m1,m2):continue
                elif len(ids)>=4:hadronic4pi+=1
                break
        primary_rows.append(a.tolist())
    # Use exHad's own terminal decay convention and 8.317 secondary decayer.
    config=ROOT/'.runtime/current.json'
    if not config.is_file():raise RuntimeError('Build exHad and run python tools/configure.py before generating decays')
    finished=decayer(json.loads(config.read_text())['xmldoc']).finish(primary_rows,seed+120903)
    axes=directions(rng,count);events=[];maxerr=0.
    for q,event,axis in zip(qs,finished,axes):
        eq=(m2*m2-m1*m1+q*q)/(2*m2);k=np.sqrt(max(0,eq*eq-q*q));frame=np.array([0,0,k,eq])
        a=np.array(event);a[:,:4]=boost(a[:,:4],frame);a[:,:3]=rotate_z(a[:,:3],np.broadcast_to(axis,(len(a),3)))
        dark=np.array([*(-k*axis),m2-eq,m1,4900101])
        a=np.vstack([dark,a]);events.append(a)
        maxerr=max(maxerr,float(np.max(abs(a[:,:4].sum(axis=0)-[0,0,0,m2]))))
    if maxerr>2e-7:raise RuntimeError(f'iDM four-momentum check failed {maxerr}')
    kmax=max(map(len,events));out=np.zeros((count,kmax,6));out[:,:,5]=-999
    for i,a in enumerate(events):out[i,:len(a)]=a
    return dict(particles=out,q=qs,channel=np.array([catalog()[i].label for i in labels]),width_per_eps2=rates.sum(),ctau_times_eps2=HBARC/rates.sum(),max_conservation_error=maxerr,phase_space_four_pion_events=hadronic4pi)


def produce(mediator_p4,mass,polarization,rng,ratio=3.,delta=.1):
    """A' -> chi1 chi2 with the vector angular law for T/L/unpolarized A'.

    polarization is the longitudinal fraction in the supplied helicity frame.
    Daughter spin is not polarized for this parity-conserving vector coupling.
    """
    p4=np.atleast_2d(mediator_p4);m1=mass/ratio;m2=(1+delta)*m1
    e2=(mass*mass+m2*m2-m1*m1)/(2*mass);k=np.sqrt(e2*e2-m2*m2)
    a=(mass*mass-(m2-m1)**2)/2;fL=np.broadcast_to(polarization,len(p4))
    dirs=np.empty((len(p4),3));left=np.arange(len(p4))
    while len(left):
        d=directions(rng,len(left));c=d[:,2]
        w=(1-fL[left])*(a-k*k*(1-c*c))+fL[left]*(a-2*k*k*c*c)
        ok=rng.random(len(left))*a<w;dirs[left[ok]]=d[ok];left=left[~ok]
    momentum=np.linalg.norm(p4[:,:3],axis=1)
    axes=np.divide(p4[:,:3],momentum[:,None],out=np.tile([0.,0.,1.],(len(p4),1)),where=momentum[:,None]>0)
    v=rotate_z(k*dirs,axes);return boost(np.column_stack([v,np.full(len(v),e2)]),p4)
