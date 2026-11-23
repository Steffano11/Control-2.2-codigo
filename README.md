# 4. Análisis de métodos computacionales para ranking en grafos dirigidos

**[5 puntos]**

Sea \(G=(V,E)\) un grafo dirigido con \(V=\{1,\ldots,n\}\) nodos y matriz de adyacencia \(A\) (\(A_{ij}=1\) si existe el enlace \(j\to i\)). Sea \(d_j\) el grado de salida del nodo \(j\), \(D=\{j:d_j=0\}\) el conjunto de nodos sin salida y \(S=A\operatorname{diag}(d)^{-1}\) con las columnas de \(D\) sustituidas por \(\frac{1}{n}e\).

Define:

$$
M_\alpha=I-\alpha S,
\qquad
M_\alpha x=\frac{1-\alpha}{n}e,
\qquad
\pi=\frac{x}{\|x\|_1,
}
$$

y sea \(M_\alpha=U\Sigma V^T\) su SVD. La solución con pseudoinversa truncada de rango \(k\) es:

$$
x_k=\sum_{i=1}^{k}\sigma_i^{-1}v_i u_i^T
\frac{1-\alpha}{n}e.
$$

Para \(\alpha\to1\), ¿qué método computacional (potencia, solución con `pinv`, pseudoinversa truncada o SVD truncada dispersa) ofrece el mejor compromiso entre tiempo, memoria y fidelidad del ranking, y cómo dependen \(\sigma_{\min}(M_\alpha)\), \(\kappa_2(M_\alpha)\) y \(k\) necesario de la estructura del grafo \(G\)?

## a) Generación de grafos

Con `networkx` y `scipy.sparse`, genere grafos Erdős–Rényi y Barabási–Albert con \(n\in\{100,500,1000,2000,5000\}\), con y sin nodos sin salida, y cargue un grafo real (por ejemplo web-Stanford o un subgrafo de SNAP).

Construya \(S\) y \(M_\alpha\) en formato disperso y denso.

## b) Implementación de cuatro métodos

1. Método de la potencia (referencia).
2. Solución exacta con `numpy.linalg.pinv`.
3. Pseudoinversa truncada programada por usted desde `numpy.linalg.svd`.
4. SVD truncada dispersa con `scipy.sparse.linalg.svds`, o SVD aleatorizada.

Valide que (a), (b) y (c) con \(k=n\) coinciden con tolerancia \(10^{-8}\).

## c) Espectro y condicionamiento

Para \(\alpha\in\{0.5,0.85,0.95,0.99,0.999\}\) grafique \(\sigma_i\) en escala logarítmica y \(\kappa_2(M_\alpha)\) vs. \(\alpha\).

Estime el rango numérico efectivo con distintas tolerancias.

## d) Error y calidad del ranking

Para cada método y \(k\) variable, mida:

- \(\|\pi-\pi_k\|_1\)
- Kendall \(\tau\)
- Precisión@\({m}\) (\(m=10,50,100\)

Determine el menor \(k\) con \(\tau>0.95\) y su escala respecto de \(n\), \(\alpha\) y la distribución de grados.

## e) Análisis por nodo

Proyecte los últimos vectores singulares sobre los nodos y detecte cuáles reciben mayor peso (sin salida, bajo grado, puentes, hubs).

Calcule \(|\pi_i-(\pi_k)_i|\) y el cambio de posición de cada nodo, y visualice el grafo con tamaño de nodo \(\propto\pi_i\) y color según el cambio de ranking.

## f) Rendimiento

Mida tiempo (`time.perf_counter`, promedio de 5 corridas) y memoria (`tracemalloc` o `memory_profiler`).

Grafique tiempo y memoria vs. \(n\) en escala log-log, estime empíricamente el orden de crecimiento de cada método y compare con el costo teórico \(O(n^3)\) para SVD densa.

## g) Robustez

Repita los experimentos con 10 semillas aleatorias, reporte media y desviación estándar, y analice la sensibilidad de \(\pi\) a perturbar el 1 % de las aristas.

## h) Conclusiones

Indique en qué regímenes (\(n\), \(\alpha\), densidad) conviene cada método y cuándo el truncamiento reordena nodos importantes.
