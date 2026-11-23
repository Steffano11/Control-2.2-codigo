## 4. Análisis de métodos computacionales para ranking en grafos

**[5 puntos]**

Sea \(G=(V,E)\) un grafo dirigido con \(V=\{1,\ldots,n\}\) nodos y matriz de adyacencia \(A\), donde \(A_{ij}=1\) si existe el enlace \(j\to i\). Sea \(d_j\) el grado de salida del nodo \(j\), \(D=\{j:d_j=0\}\) el conjunto de nodos sin salida y \(S=A\operatorname{diag}(d)^{-1}\), con las columnas de \(D\) sustituidas por \(\frac{1}{n}\mathbf{e}\).

Defina:

$$
M_\alpha=I-\alpha S,
\qquad
M_\alpha x=\frac{1-\alpha}{n}\mathbf{e},
\qquad
\pi=\frac{x}{\lVert x\rVert_1.
}
$$

Sea \(M_\alpha=U\Sigma V^\top\) su descomposición en valores singulares (SVD). La solución con pseudoinversa truncada de rango \(k\) es:

$$
x_k=\sum_{i=1}^{k}
\sigma_i^{-1}v_i u_i^\top
\left(\frac{1-\alpha}{n}\mathbf{e}\right).
$$

Para \(\alpha\to1\), ¿qué método computacional (potencia, solución con `pinv`, pseudoinversa truncada o SVD truncada dispersa) ofrece el mejor compromiso entre tiempo, memoria y fidelidad del ranking, y cómo dependen \(\sigma_{\min}(M_\alpha)\), \(\kappa_2(M_\alpha)\) y el \(k\) necesario de la estructura del grafo \(G\)?

### a) Generación de grafos

Con `networkx` y `scipy.sparse`, genere grafos Erdős–Rényi y Barabási–Albert con \(n\in\{100,500,1000,2000,5000\}\), con y sin nodos sin salida, y cargue un grafo real (por ejemplo, web-Stanford o un subgrafo de SNAP).

Construya \(S\) y \(M_\alpha\) en formato disperso y denso.

### b) Implementación de cuatro métodos

1. Método de la potencia (referencia).
2. Solución exacta con `numpy.linalg.pinv`.
3. Pseudoinversa truncada programada desde `numpy.linalg.svd`.
4. SVD truncada dispersa con `scipy.sparse.linalg.svds` o SVD aleatorizada.

Valide que los métodos (a), (b) y (c), con \(k=n\), coincidan con una tolerancia de \(10^{-8}\).

### c) Espectro y condicionamiento

Para \(\alpha\in\{0.5,0.85,0.95,0.99,0.999\}\), grafique \(\sigma_i\) en escala logarítmica y \(\kappa_2(M_\alpha)\) vs. \(\alpha\).

Estime el rango numérico efectivo con distintas tolerancias.

### d) Error y calidad del ranking

Para cada método y \(k\) variable, mida:

- Error de norma \(L_1\):

  $$\lVert\pi-\pi_k\rVert_1$$

- Correlación de Kendall:

  $$\tau(\pi,\pi_k)$$

- Precisión@\(m\), con \(m\in\{10,50,100\}\).

Determine el menor \(k\) tal que:

$$
\tau(\pi,\pi_k)>0.95
$$

y analice su escala respecto de \(n\), \(\alpha\) y la distribución de grados.

### e) Análisis por nodo

Proyecte los últimos vectores singulares sobre los nodos y detecte cuáles reciben mayor peso (sin salida, bajo grado, puentes, hubs).

Calcule:

$$
\left|\pi_i-(\pi_k)_i\right|
$$

y el cambio de posición de cada nodo. Visualice el grafo con tamaño de nodo proporcional a \(\pi_i\) y color según el cambio de ranking.

### f) Rendimiento

Mida tiempo (`time.perf_counter`, promedio de 5 corridas) y memoria (`tracemalloc` o `memory_profiler`).

Grafique tiempo y memoria vs. \(n\) en escala log-log, estime empíricamente el orden de crecimiento de cada método y compare con el costo teórico \(O(n^3)\) para SVD densa.

### g) Robustez

Repita los experimentos con 10 semillas aleatorias, reporte media y desviación estándar, y analice la sensibilidad de \(\pi\) a perturbar el 1 % de las aristas.

### h) Conclusiones

Indique en qué regímenes (\(n\), \(\alpha\), densidad) conviene cada método y cuándo el truncamiento reordena nodos importantes.
