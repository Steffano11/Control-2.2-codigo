
from __future__ import annotations
import argparse
import csv
import gzip
import json
import math
import os
import time
import tracemalloc
import warnings
from pathlib import Path

# Evita sobreejecucion de bibliotecas BLAS en experimentos pequenos.
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, svds
from scipy.stats import kendalltau

BASE = Path(__file__).resolve().parent
OUT = BASE / 'resultados'
OUT.mkdir(exist_ok=True)
ALPHAS = [0.5, 0.85, 0.95, 0.99, 0.999]


def generate_graph(kind: str, n: int, seed: int, dangling: bool, m: int = 3):
    """Genera grafo dirigido. BA se orienta aleatoriamente, NO bidireccional.
    'dangling' elimina enlaces de salida del 10%% de los nodos; sin dangling
    se garantiza al menos una arista de salida por nodo.
    """
    rng = np.random.default_rng(seed)
    if kind == 'ER':
        g = nx.gnp_random_graph(n, min(1., 4/(n-1)), seed=seed, directed=True)
    elif kind == 'BA':
        und = nx.barabasi_albert_graph(n, m, seed=seed)
        g = nx.DiGraph()
        g.add_nodes_from(range(n))
        for u, v in und.edges():
            if rng.integers(2): g.add_edge(u, v)
            else: g.add_edge(v, u)
    else:
        raise ValueError('tipo desconocido')
    if dangling:
        victims = rng.choice(n, size=max(1, n//10), replace=False)
        for j in victims: g.remove_edges_from(list(g.out_edges(int(j))))
    else:
        for j, degree in list(g.out_degree()):
            if degree == 0:
                i = int(rng.integers(n-1))
                if i >= j: i += 1
                g.add_edge(j, i)
    return g


def load_snap(path: str, max_nodes: int = 5000):
    """Lee archivo SNAP edge list, seleccionando un subgrafo inducido.
    Seleccion: primeros max_nodes identificadores ordenados. Reportar sesgo.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError('Descargue Wiki-Vote.txt.gz de https://snap.stanford.edu/data/wiki-Vote.html')
    opener = gzip.open if p.suffix.lower() == '.gz' else open
    edges = []
    nodes = set()
    with opener(p, 'rt', encoding='utf8') as f:
        for line in f:
            if not line.strip() or line.lstrip().startswith('#'): continue
            src, dst = map(int, line.split()[:2])
            edges.append((src, dst))
            nodes.update([src, dst])
    chosen = sorted(nodes)[:max_nodes]
    lookup = {node: i for i, node in enumerate(chosen)}
    g = nx.DiGraph()
    g.add_nodes_from(range(len(chosen)))
    g.add_edges_from((lookup[a], lookup[b]) for a,b in edges if a in lookup and b in lookup and a!=b)
    return g, {'original_nodes':len(nodes), 'original_edges':len(edges), 'sample_nodes':g.number_of_nodes(), 'sample_edges':g.number_of_edges(), 'sampling':'first sorted node IDs, induced subgraph'}


def adjacency(g):
    n = g.number_of_nodes()
    # Coordenadas (destino, origen); no invertir esta convencion.
    a = [(dst,src) for src,dst in g.edges() if src!=dst]
    if not a: return sp.csc_matrix((n,n), dtype=float)
    rr,cc = np.array(a,dtype=int).T
    mat = sp.coo_matrix((np.ones(len(rr)),(rr,cc)),shape=(n,n)).tocsc()
    mat.data[:] = 1.0
    return mat


class PageRankMatrix:
    """S=P+e d^T/n, P=A diag(1/deg) con inversa 0 en grado 0.
    Evita materializar columnas densas por nodos dangling en n grande.
    """
    def __init__(self, a):
        self.A = a.tocsc().astype(float)
        self.n = a.shape[0]
        deg = np.asarray(self.A.sum(axis=0)).ravel()
        self.dangling = deg == 0
        inv = np.divide(1.0, deg, out=np.zeros_like(deg), where=deg>0)
        self.P = (self.A @ sp.diags(inv,format='csc')).tocsc()
        self.ones = np.ones(self.n, dtype=float)

    def S_times(self, v):
        if np.ndim(v) == 1:
            return self.P @ v + (np.sum(v[self.dangling])/self.n)*self.ones
        return self.P @ v + np.outer(self.ones/self.n, np.sum(v[self.dangling,:],axis=0))

    def ST_times(self, v):
        if np.ndim(v) == 1:
            return self.P.T @ v + self.dangling.astype(float)*(np.sum(v)/self.n)
        return self.P.T @ v + np.outer(self.dangling.astype(float)/self.n, np.sum(v,axis=0))

    def M_op(self, alpha):
        n=self.n
        return LinearOperator((n,n),
            matvec=lambda v: v-alpha*self.S_times(v),
            rmatvec=lambda v: v-alpha*self.ST_times(v),dtype=float)

    def S_dense(self):
        if self.n > 1500: raise MemoryError('S densa deshabilitada para n>1500')
        return self.P.toarray()+np.outer(self.ones/self.n,self.dangling.astype(float))

    def M_dense(self,alpha):
        return np.eye(self.n)-alpha*self.S_dense()

    def M_sparse(self,alpha,explicit=True):
        """Matriz CSC exacta (costo n*dangling) o LinearOperator implícito."""
        if not explicit: return self.M_op(alpha)
        s=self.P
        if np.any(self.dangling):
            cols=np.flatnonzero(self.dangling)
            r=np.tile(np.arange(self.n),len(cols))
            c=np.repeat(cols,self.n)
            add=sp.coo_matrix((np.full(len(r),1/self.n),(r,c)),shape=(self.n,self.n)).tocsc()
            s=s+add
        return (sp.eye(self.n,format='csc')-alpha*s).tocsc()


def page_rank_power(pm, alpha, tol=1e-12, max_iter=50000):
    n=pm.n
    x=np.full(n,1/n)
    b=(1-alpha)/n
    for iteration in range(1,max_iter+1):
        y=alpha*pm.S_times(x)+b
        if np.linalg.norm(y-x,ord=1)<tol:
            return y/y.sum(),iteration
        x=y
    warnings.warn(f'Metodo potencia no convergio: alpha={alpha}, iter={max_iter}')
    return x/x.sum(),max_iter


def pi_from_x(x):
    z=np.real_if_close(x).astype(float)
    # No aplicar valor absoluto sin explicarlo; resultados truncados pueden ser negativos.
    if z.sum() == 0: return np.full(len(z),np.nan)
    return z/np.linalg.norm(z,ord=1)


def page_rank_pinv(pm,alpha,rcond=1e-14):
    b=np.full(pm.n,(1-alpha)/pm.n)
    return pi_from_x(np.linalg.pinv(pm.M_dense(alpha),rcond=rcond) @ b)


def manual_truncated_from_svd(u,s,vh,b,k):
    # x_k = sum_{i=1}^k (u_i^T b / sigma_i) v_i.
    # CUIDADO: se usan los MAYORES valores singulares, como indica el enunciado.
    k=min(int(k),len(s))
    vec=vh[:k,:].T @ ((u[:,:k].T @ b)/s[:k])
    return pi_from_x(vec)


def sparse_truncated(pm,alpha,k,tol=1e-8):
    if k >= pm.n: raise ValueError('svds exige k<n; para k=n usar SVD densa')
    op=pm.M_op(alpha)
    u,s,vt=svds(op,k=k,which='LM',tol=tol,maxiter=2500,return_singular_vectors=True,
                  solver='arpack',random_state=123)
    order=np.argsort(s)[::-1]
    u,s,vt=u[:,order],s[order],vt[order,:]
    b=np.full(pm.n,(1-alpha)/pm.n)
    return manual_truncated_from_svd(u,s,vt,b,k)


def ranking_metrics(reference,approx):
    n=len(reference)
    if not np.isfinite(approx).all():
        return {'l1':np.nan,'tau':np.nan,**{f'p{m}':np.nan for m in (10,50,100)}}
    tau=kendalltau(reference,approx,variant='b').statistic
    out={'l1':float(np.linalg.norm(reference-approx,1)),'tau':float(tau)}
    for m in [10,50,100]:
        m=min(n,m)
        actual=set(np.argsort(-reference,kind='stable')[:m]); pred=set(np.argsort(-approx,kind='stable')[:m])
        out[f'p{m}']=len(actual&pred)/m
    return out


def save_rows(name,rows):
    if not rows: return
    import pandas as pd
    pd.DataFrame(rows).to_csv(OUT/name,index=False)


def timing(fn,repeats=5):
    values=[]; mems=[]; res=None
    import gc
    for _ in range(repeats):
        gc.collect()
        tracemalloc.start()
        t=time.perf_counter()
        res=fn()
        dt=time.perf_counter()-t
        _,peak=tracemalloc.get_traced_memory()
        tracemalloc.stop()
        values.append(dt); mems.append(peak/1024**2)
    return res,float(np.mean(values)),float(np.std(values,ddof=1)) if repeats>1 else 0.0,float(np.mean(mems))


def estimate_spectrum(pm,alpha):
    m=pm.M_dense(alpha)
    vals=np.linalg.svd(m,compute_uv=False)
    return vals


def plt_setup():
    plt.rcParams.update({'font.size':9, 'axes.spines.top':False,
                         'axes.spines.right':False,'figure.dpi':120,'savefig.dpi':170})


def draw_network(g,base,approx,pm,title,path):
    n=pm.n
    # cap graph display for readability
    selected=np.argsort(-base)[:min(n,90)]
    sub=g.subgraph(selected).copy()
    pos=nx.spring_layout(sub,seed=37)
    nodes=list(sub.nodes())
    delta=np.array([np.where(np.argsort(-approx)==node)[0][0]-np.where(np.argsort(-base)==node)[0][0] for node in nodes])
    sizes=250+8500*np.array([base[node] for node in nodes])
    fig,ax=plt.subplots(figsize=(8,6))
    nx.draw_networkx_edges(sub,pos,alpha=.16,arrows=False,width=.45,ax=ax)
    coll=nx.draw_networkx_nodes(sub,pos,node_size=sizes,node_color=delta,cmap='coolwarm',ax=ax,vmin=int(delta.min()),vmax=int(delta.max()),linewidths=.2,edgecolors='#444')
    fig.colorbar(coll,ax=ax,shrink=.68,label='Cambio de puesto: rango truncado - rango referencia')
    ax.set_title(title); ax.set_axis_off(); fig.tight_layout(); fig.savefig(path);plt.close(fig)


def run(args):
    plt_setup()
    params={'modo':args.modo,'semilla':args.seed,'snap':args.snap,'timestamp_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()}
    nlist=[100,500,1000,2000,5000]
    records=[]; graph_rows=[]; spectra=[]; ranks=[]; perf=[]; node_rows=[]; stability=[]; fullrank=[]
    available=[]
    sample = generate_graph('ER', 100, args.seed, True)
    for kind in ['ER','BA']:
      for dangling in [False,True]:
        for n in nlist:
          g=generate_graph(kind,n,args.seed,dangling)
          a=adjacency(g); pm=PageRankMatrix(a)
          graph_rows.append({'tipo':kind,'n':n,'sin_salida_forzados':dangling,
                             'aristas':g.number_of_edges(),'densidad':g.number_of_edges()/(n*(n-1)),
                             'nodos_sin_salida':int(pm.dangling.sum()),'nnz_P':pm.P.nnz,
                             'bytes_P_csc':int(pm.P.data.nbytes+pm.P.indices.nbytes+pm.P.indptr.nbytes),
                             'bytes_M_densa_estimados':int(n*n*8),
                             'bytes_dangling_explicitos_estimados':int(n*int(pm.dangling.sum())*12)})
          # En modo rapido, los mas grandes SOLO se miden con potencia.
          result,t,sd,mem=timing(lambda:page_rank_power(pm,.85),5)
          perf.append({'tipo':kind,'n':n,'dangling':dangling,'metodo':'potencia','alpha':.85,
                       'k':0,'tiempo_s':t,'sd_tiempo_s':sd,'peak_tracemalloc_mib':mem,
                       'iteraciones':result[1]})
          # Restriccion computacional explicita, no mediciones inventadas.
          if kind=='ER' and not dangling:
            for k in ([5,15] if n<=500 else [5]):
              try:
                _,t,sd,mem=timing(lambda:sparse_truncated(pm,.85,k),5)
                perf.append({'tipo':kind,'n':n,'dangling':dangling,'metodo':'svds (LM)',
                             'alpha':.85,'k':k,'tiempo_s':t,'sd_tiempo_s':sd,'peak_tracemalloc_mib':mem,
                             'iteraciones':np.nan})
              except Exception as ex:
                available.append(f'svds {kind} n={n} k={k}: {type(ex).__name__}: {ex}')
          if n==100 and kind in ('ER','BA') and dangling in (False,True):
            # Prueba de igualdad entre implementacion densa, CSC y operador implicito.
            alpha_test=.85
            rng_test=np.random.default_rng(2026)
            v_test=rng_test.normal(size=n)
            np.testing.assert_allclose(pm.M_dense(alpha_test) @ v_test,
                                       pm.M_sparse(alpha_test) @ v_test,
                                       rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(pm.M_dense(alpha_test) @ v_test,
                                       pm.M_op(alpha_test) @ v_test,
                                       rtol=1e-12,atol=1e-12)
            for alpha in ALPHAS:
              ref,it=page_rank_power(pm,alpha)
              denseM=pm.M_dense(alpha)
              # Un unico SVD completo por (grafo, alpha); pinv se ejecuta separadamente.
              u,s,vh=np.linalg.svd(denseM,full_matrices=False)
              pinv_res=page_rank_pinv(pm,alpha)
              full_manual=manual_truncated_from_svd(u,s,vh,np.full(n,(1-alpha)/n),n)
              fullrank.append({'tipo':kind,'dangling':dangling,'n':n,'alpha':alpha,
                'max_l1_pinv_potencia':np.linalg.norm(ref-pinv_res,1),
                'max_l1_svd_completa_potencia':np.linalg.norm(ref-full_manual,1),
                'iteraciones_potencia':it})
              # El titulo dice validacion 1e-8; guardamos error si no la supera.
              for i,si in enumerate(s):
                spectra.append({'tipo':kind,'dangling':dangling,'n':n,'alpha':alpha,'idx':i+1,'sigma':float(si),
                                 'sigma_min':float(s[-1]),'sigma_max':float(s[0]),
                                 'kappa':float(s[0]/s[-1]),
                                 'r_eff_1e-2':int((s>s[0]*1e-2).sum()),
                                 'r_eff_1e-4':int((s>s[0]*1e-4).sum()),
                                 'r_eff_1e-6':int((s>s[0]*1e-6).sum())})
              kvals=[5,10,20,40,60,80,90,95,99,100]
              kvals=sorted(set(kvals))
              for k in kvals:
                pred=manual_truncated_from_svd(u,s,vh,np.full(n,(1-alpha)/n),k)
                ranks.append({'tipo':kind,'dangling':dangling,'n':n,'alpha':alpha,'metodo':'SVD truncada densa','k':k,
                              **ranking_metrics(ref,pred)})
                if k in [5,20,60]:
                  try:
                    spr=sparse_truncated(pm,alpha,k)
                    ranks.append({'tipo':kind,'dangling':dangling,'n':n,'alpha':alpha,
                                  'metodo':'svds dispersa LM','k':k,**ranking_metrics(ref,spr)})
                  except Exception as ex:
                    available.append(f'error svds ranking {kind} {alpha} {k}: {ex}')
              if alpha==.99 and kind=='BA' and dangling:
                rank_approx=manual_truncated_from_svd(u,s,vh,np.full(n,(1-alpha)/n),80)
                first=np.argsort(-ref,kind='stable');new=np.argsort(-rank_approx,kind='stable')
                oldrank=np.empty(n,int);newrank=np.empty(n,int)
                oldrank[first]=np.arange(1,n+1);newrank[new]=np.arange(1,n+1)
                roles={'dangling':pm.dangling,'hub':np.array([g.in_degree(i) for i in range(n)])>=np.percentile([g.in_degree(i) for i in range(n)],90),
                       'bridge':np.array([g.degree(i) for i in range(n)])>=np.percentile([g.degree(i) for i in range(n)],90)}
                for i in range(n):
                  node_rows.append({'nodo':i,'pi_ref':ref[i],'pi_k80':rank_approx[i],
                    'error':abs(ref[i]-rank_approx[i]),'rank_ref':oldrank[i],'rank_k80':newrank[i],
                    'cambio_rango':newrank[i]-oldrank[i],'in_degree':g.in_degree(i),
                    'out_degree':g.out_degree(i),'dangling':bool(pm.dangling[i]),
                    'u_ultimo':u[i,-1],'v_ultimo':vh[-1,i]})
                draw_network(g,ref,rank_approx,pm,'BA dirigido, n=100, alpha=0.99, k=80',OUT/'grafo_ranking.png')
    # Comparacion exacta y tiempo para numpy.pinv (n=100); costo no extrapolado a n=5000.
    for n in [100,500] + ([1000] if args.modo=='completo' else []):
        pm=PageRankMatrix(adjacency(generate_graph('ER',n,args.seed,False)))
        for name,fn in [('pinv',lambda:page_rank_pinv(pm,.85)),
                        ('SVD densa completa',lambda:np.linalg.svd(pm.M_dense(.85),full_matrices=False))]:
            _,t,sd,mem=timing(fn,5)
            perf.append({'tipo':'ER','n':n,'dangling':False,'metodo':name,'alpha':.85,
                         'k':n,'tiempo_s':t,'sd_tiempo_s':sd,'peak_tracemalloc_mib':mem,'iteraciones':np.nan})
    # Robustez 10 semillas, 1% de aristas alteradas (eliminadas y reemplazadas).
    rng_seed=args.seed
    for kind in ['ER','BA']:
      for dangling in [False,True]:
       for alpha in [.85,.99]:
        for seed in range(rng_seed,rng_seed+10):
          g=generate_graph(kind,100,seed,dangling)
          pm=PageRankMatrix(adjacency(g))
          ref,_=page_rank_power(pm,alpha)
          # muestreo uniformemente de aristas existentes; aleatoriedad local reproducible.
          rng=np.random.default_rng(seed+500000)
          pert=g.copy(); edges=list(pert.edges())
          q=max(1,round(.01*len(edges)))
          for idx in rng.choice(len(edges),size=min(q,len(edges)),replace=False):
            pert.remove_edge(*edges[int(idx)])
          added=0; limit=0
          while added<q and limit<100*q+100:
            a,b=rng.integers(0,100,size=2);limit+=1
            if a!=b and not pert.has_edge(int(a),int(b)):
              pert.add_edge(int(a),int(b));added+=1
          newref,_=page_rank_power(PageRankMatrix(adjacency(pert)),alpha)
          # Truncada en original k=80
          u,s,vh=np.linalg.svd(pm.M_dense(alpha),full_matrices=False)
          approx=manual_truncated_from_svd(u,s,vh,np.full(pm.n,(1-alpha)/pm.n),80)
          stability.append({'tipo':kind,'dangling':dangling,'alpha':alpha,'seed':seed,
                            'aristas_originales':len(edges),'aristas_perturbadas':q,
                            'l1_perturbacion':float(np.linalg.norm(ref-newref,1)),
                            'tau_perturbacion':float(kendalltau(ref,newref).statistic),
                            'tau_truncacion':float(kendalltau(ref,approx).statistic),
                            'l1_truncacion':float(np.linalg.norm(ref-approx,1))})
    # Escalamiento de k: experimento mas costoso, solo en modo completo.
    # n=500: ambas familias, con/sin colgantes, alpha .85 y .99.
    # n=1000: ER y BA sin colgantes, alpha .99.
    if args.modo=='completo':
      for n in [500,1000]:
       combinations=[('ER',False),('ER',True),('BA',False),('BA',True)] if n==500 else [('ER',False),('BA',False)]
       for kind,dangling in combinations:
        g=generate_graph(kind,n,args.seed,dangling)
        pm=PageRankMatrix(adjacency(g))
        for alpha in ([.85,.99] if n==500 else [.99]):
         reference,_=page_rank_power(pm,alpha)
         u,s,vh=np.linalg.svd(pm.M_dense(alpha),full_matrices=False)
         b=np.full(n,(1-alpha)/n)
         for k in sorted(set([5,20,round(.25*n),round(.5*n),round(.8*n),round(.95*n),n-1,n])):
           approximation=manual_truncated_from_svd(u,s,vh,b,k)
           ranks.append({'tipo':kind,'dangling':dangling,'n':n,'alpha':alpha,
                         'metodo':'SVD truncada densa','k':k,
                         **ranking_metrics(reference,approximation)})
    # SNAP ejecutable solo con archivo real. No sustituir silenciosamente por ER.
    if args.snap:
      g,details=load_snap(args.snap,min(args.snap_max,5000))
      pm=PageRankMatrix(adjacency(g)); ref,it=page_rank_power(pm,.85)
      np.savetxt(OUT/'snap_pagerank.csv',np.column_stack([np.arange(pm.n),ref]),delimiter=',',
                 header='nodo_local,pi',comments='')
      params['snap_informacion']=details
      params['snap_iteraciones']=it
      params['snap_ejecutado']=True
    else:
      params['snap_ejecutado']=False
      params['snap_justificacion']='No se proporciono archivo real; requiere descargarlo de SNAP.'
    # Salvar datos
    for name,rowset in [('grafos.csv',graph_rows),('espectro.csv',spectra),('ranking.csv',ranks),
                        ('rendimiento.csv',perf),('analisis_nodos.csv',node_rows),
                        ('robustez.csv',stability),('validacion.csv',fullrank)]:
       save_rows(name,rowset)
    (OUT/'metadatos.json').write_text(json.dumps({'parametros':params,'advertencias':available},indent=2,ensure_ascii=False),encoding='utf8')
    plot_results(graph_rows,spectra,ranks,perf,stability)
    import pandas as pd
    nd = pd.DataFrame(node_rows)
    if len(nd):
        q=nd.assign(peso_v=np.abs(nd.v_ultimo)).nlargest(12,'peso_v').sort_values('peso_v')
        fig,ax=plt.subplots(figsize=(7,2.6))
        ax.barh(q.nodo.astype(str),q.peso_v,
                color=['#9c6874' if b else '#4478a8' for b in q.dangling])
        ax.set(xlabel='Magnitud |v_min(i)|',ylabel='Nodo',
               title='Doce nodos con mayor peso en el ultimo vector singular')
        fig.tight_layout();fig.savefig(OUT/'vectores_singulares.png');plt.close(fig)
    rankings=pd.DataFrame(ranks)
    best=[]
    for (kind,dangling,n,alpha),part in rankings[rankings.metodo=='SVD truncada densa'].groupby(['tipo','dangling','n','alpha']):
      found=part[part.tau>0.95]
      best.append({'tipo':kind,'dangling':dangling,'n':n,'alpha':alpha,
                    'k_min_tau_gt_095':int(found.k.min()) if len(found) else None,
                    'k_minprobados':int(part.k.min()),'k_maxprobados':int(part.k.max())})
    save_rows('k_optimo.csv',best)
    print('\nVALIDACION EXACTA:')
    for row in fullrank[:5]: print(row)
    print('\nTABLA K MINIMO:')
    for row in best: print(row)
    print('\nRESULTADOS:',OUT)


def plot_results(graph_rows,spectra,ranks,perf,stability):
    import pandas as pd
    spectra=pd.DataFrame(spectra);ranks=pd.DataFrame(ranks);perf=pd.DataFrame(perf)
    stability=pd.DataFrame(stability)
    spec=spectra[(spectra.tipo=='ER') & (spectra.dangling==False)]
    fig,ax=plt.subplots(figsize=(7,4))
    for a in ALPHAS:
        p=spec[spec.alpha==a]
        ax.semilogy(p.idx,p.sigma,label=f'alpha={a}',linewidth=1.5)
    ax.set(xlabel='Indice singular (mayor a menor)',ylabel='Valor singular',title='Espectro de M en ER, n=100')
    ax.legend(fontsize=7);fig.tight_layout();fig.savefig(OUT/'espectro.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.5,4))
    for kind,ls in [('ER','-'),('BA','--')]:
      for d,marker in [(False,'o'),(True,'s')]:
        p=spectra[(spectra.tipo==kind)&(spectra.dangling==d)&(spectra.idx==1)]
        ax.semilogy(p.alpha,p.kappa,linestyle=ls,marker=marker,label=f'{kind}, dangling={d}')
    ax.set(xlabel='alpha',ylabel='Condicionamiento kappa_2',title='Condicionamiento vs. alpha (n=100)')
    ax.legend(fontsize=7);fig.tight_layout();fig.savefig(OUT/'condicionamiento.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    p=ranks[(ranks.tipo=='ER')&(ranks.dangling==False)&(ranks.metodo=='SVD truncada densa')]
    for a in ALPHAS:
        q=p[p.alpha==a];ax.plot(q.k,q.tau,marker='.',label=f'alpha={a}')
    ax.axhline(.95,color='gray',linestyle=':',label='Objetivo tau=0.95')
    ax.set(xlabel='Rango k',ylabel='Kendall tau',ylim=(-1.1,1.05),title='Fidelidad del ordenamiento segun k')
    ax.legend(fontsize=7);fig.tight_layout();fig.savefig(OUT/'tau_vs_k.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for a in ALPHAS:
        q=p[p.alpha==a];ax.semilogy(q.k,np.maximum(q.l1,1e-16),marker='.',label=f'alpha={a}')
    ax.set(xlabel='Rango k',ylabel='Error L1',title='Error de PageRank frente a k (ER, n=100)')
    ax.legend(fontsize=7);fig.tight_layout();fig.savefig(OUT/'error_vs_k.png');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(9,3.6))
    for method in ['potencia','svds (LM)','pinv','SVD densa completa']:
      q=perf[(perf.tipo=='ER')&(perf.dangling==False)&(perf.metodo==method)]
      if len(q):
        q=q.sort_values('n')
        axs[0].loglog(q.n,q.tiempo_s,marker='o',label=method)
        axs[1].loglog(q.n,np.maximum(q.peak_tracemalloc_mib,1e-3),marker='o',label=method)
    axs[0].set(xlabel='n',ylabel='Segundos (media de 5)',title='Tiempo medido')
    axs[1].set(xlabel='n',ylabel='MiB trazados (media de 5)',title='Memoria (tracemalloc)')
    axs[0].legend(fontsize=7);axs[1].legend(fontsize=7)
    fig.tight_layout();fig.savefig(OUT/'rendimiento.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,3.6))
    for kind in ['ER','BA']:
        for d in [False,True]:
            q=stability[(stability.tipo==kind)&(stability.dangling==d)].groupby('alpha').tau_perturbacion.agg(['mean','std']).reset_index()
            ax.errorbar(q.alpha,q['mean'],q['std'],marker='o',capsize=3,label=f'{kind}, dangling={d}')
    ax.set(xlabel='alpha',ylabel='Kendall tau (original vs. 1% perturbado)',ylim=(-1,1.05),title='Robustez: 10 semillas')
    ax.legend(fontsize=7);fig.tight_layout();fig.savefig(OUT/'robustez.png');plt.close(fig)
    # escala teorica memoria: n^2 para matrices densas, O(n+m) para CSC sin rank-one
    dat=pd.DataFrame(graph_rows)
    fig,ax=plt.subplots(figsize=(7,3.6))
    for kind in ['ER','BA']:
        q=dat[(dat.tipo==kind)&(dat.sin_salida_forzados==False)]
        ax.loglog(q.n,q.bytes_P_csc/1024**2,marker='o',label=f'P CSC {kind}')
    q=dat[(dat.tipo=='ER')&(dat.sin_salida_forzados==False)]
    ax.loglog(q.n,q.bytes_M_densa_estimados/1024**2,marker='x',label='M densa (1 matriz, estimado)')
    ax.set(xlabel='n',ylabel='MiB de almacenamiento',title='Costos estructurales de memoria')
    ax.legend(fontsize=7);fig.tight_layout();fig.savefig(OUT/'memoria_estructural.png');plt.close(fig)


if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--modo',choices=['rapido','completo'],default='rapido')
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--snap',type=str,default=None,help='Ruta .txt/.txt.gz del archivo SNAP autentico')
    ap.add_argument('--snap-max',type=int,default=5000)
    run(ap.parse_args())
