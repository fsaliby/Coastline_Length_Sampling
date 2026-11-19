"""Build the English and Portuguese teaching notebooks from one source.

    python notebooks/build_notebooks.py

writes notebooks/coastline_notebook_en.ipynb and notebooks/coastline_notebook_pt.ipynb.
Edit this file, not the notebooks: the two languages then stay identical in code.

Markdown cells are given as (english, portuguese) pairs. Code cells are shared; any
text inside <<english||portuguese>> is replaced by the version of each language.
"""

from __future__ import annotations

import re
from pathlib import Path

import nbformat as nbf

REPO = "fsaliby/Coastline_Length_Sampling"
HERE = Path(__file__).resolve().parent
CELLS: list[tuple] = []


def md(en: str, pt: str) -> None:
    CELLS.append(("md", en.strip("\n"), pt.strip("\n")))


def code(src: str) -> None:
    CELLS.append(("code", src.strip("\n")))


def badge(lang: str) -> str:
    url = (f"https://colab.research.google.com/github/{REPO}/blob/main/"
           f"notebooks/coastline_notebook_{lang}.ipynb")
    return f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({url})"


# =============================================================================== 0
md(rf"""
# How long is a coastline? Two estimators and a lesson in data analysis

{badge("en")}

This notebook measures the length of a coastline in two ways and uses the results to
introduce ideas that apply to any experiment: estimators and their sampling
distributions, power-law fits, Monte-Carlo propagation of uncertainty, statistical
versus systematic uncertainty, and validation against known answers.

- **Random points (RP).** Drop $n_p$ points at random along the coast, join them in
  order with straight segments and add up the lengths.
- **Yardstick (YS).** Walk along the coast with a ruler of fixed length $d$, as Lewis
  Fry Richardson did on maps (published in 1961).

Both lengths grow as the resolution gets finer. For many coastlines the growth follows
a power law whose exponent defines a *fractal dimension* $D$.

**How to use.** Run the cells in order (*Runtime → Run all* in Colab); the whole
notebook takes about a minute. Questions marked **Predict** are best answered before
running the next cell; click *Answer* to compare.

**Simplifications.** The Earth is treated as flat, which is a very good approximation
for coasts up to a few hundred kilometres (see Exercise 2), and the coastline is the
polyline through its points (linear interpolation). The full analysis code, on the
sphere and with more options, is in the
[repository](https://github.com/{REPO}).
""", rf"""
# Qual é o comprimento de uma linha costeira? Dois estimadores e uma lição de análise de dados

{badge("pt")}

Este notebook mede o comprimento de uma linha costeira de duas maneiras e usa os
resultados para introduzir ideias que valem para qualquer experimento: estimadores e
suas distribuições amostrais, ajuste de leis de potência, propagação de incerteza por
Monte Carlo, incerteza estatística versus sistemática e validação com respostas
conhecidas.

- **Pontos aleatórios (RP).** Sorteamos $n_p$ pontos ao longo da costa, ligamos esses
  pontos em ordem com segmentos retos e somamos os comprimentos.
- **Régua (YS, *yardstick*).** Caminhamos pela costa com uma régua de comprimento fixo
  $d$, como fez Lewis Fry Richardson com mapas (publicado em 1961).

Os dois comprimentos crescem quando a resolução fica mais fina. Para muitas costas
esse crescimento segue uma lei de potência, cujo expoente define uma *dimensão
fractal* $D$.

**Como usar.** Rode as células em ordem (*Ambiente de execução → Executar tudo* no
Colab); o notebook inteiro leva cerca de um minuto. As perguntas marcadas com
**Preveja** devem ser respondidas antes de rodar a célula seguinte; clique em
*Resposta* para comparar.

**Simplificações.** A Terra é tratada como plana, uma ótima aproximação para costas de
até algumas centenas de quilômetros (veja o Exercício 2), e a linha costeira é a
poligonal que passa pelos seus pontos (interpolação linear). O código completo da
análise, na esfera e com mais opções, está no
[repositório](https://github.com/{REPO}).
""")

# =============================================================================== 1
md(r"""
## 1. Setup
Libraries, a few constants and a fixed random seed, so that every run gives the same
numbers.
""", r"""
## 1. Preparação
Bibliotecas, algumas constantes e uma semente aleatória fixa, para que cada execução
dê os mesmos números.
""")
code(rf"""
import io
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

plt.rcParams.update({{"figure.figsize": (7.5, 4.5), "figure.dpi": 110,
                     "axes.grid": True, "grid.alpha": 0.3}})
BLUE, ORANGE, GREEN, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"

R_EARTH = 6371.0                          # <<Earth radius||raio da Terra>>, km
REPO = "https://raw.githubusercontent.com/{REPO}/main/"
rng = np.random.default_rng(2026)         # <<fixed seed||semente fixa>>


def plain(axis):
    # <<log axis labelled 20, 30 instead of 2×10¹, 3×10¹||eixo log com 20, 30 em vez de 2×10¹, 3×10¹>>
    axis.set_major_formatter(FormatStrFormatter("%g"))
    axis.set_minor_formatter(FormatStrFormatter("%g"))
""")

# =============================================================================== 2
md(r"""
## 2. The coastline

Choose the data with `SOURCE`:

- `"buzios"` (default): the Búzios peninsula, Brazil, 38 km;
- `"britain"`: the west coast of Britain, from Land's End to Cape Wrath (GSHHG
  shoreline database, 5470 km; about 10 s more to run);
- `"upload"`: your own file, either a KML with a single line (for example drawn in
  Google My Maps and exported as KML) or a text file with two columns of degrees.
  Set `UPLOAD_FORMAT` to `"kml"`, `"latlon"` or `"lonlat"`.

The points are then projected onto a plane (*flat Earth*), around the mean latitude
$\bar\varphi$:
$$x = R\,(\lambda-\lambda_0)\cos\bar\varphi, \qquad y = R\,(\varphi-\varphi_0),$$
with latitude $\varphi$ and longitude $\lambda$ in radians and $R = 6371$ km.
""", r"""
## 2. A linha costeira

Escolha os dados com `SOURCE`:

- `"buzios"` (padrão): a península de Búzios, RJ, 38 km;
- `"britain"`: a costa oeste da Grã-Bretanha, de Land's End a Cape Wrath (base de
  dados GSHHG, 5470 km; leva uns 10 s a mais);
- `"upload"`: o seu próprio arquivo, que pode ser um KML com uma única linha (por
  exemplo desenhada no Google My Maps e exportada como KML) ou um arquivo de texto com
  duas colunas em graus. Ajuste `UPLOAD_FORMAT` para `"kml"`, `"latlon"` ou `"lonlat"`.

Os pontos são então projetados num plano (*Terra plana*), em torno da latitude média
$\bar\varphi$:
$$x = R\,(\lambda-\lambda_0)\cos\bar\varphi, \qquad y = R\,(\varphi-\varphi_0),$$
com latitude $\varphi$ e longitude $\lambda$ em radianos e $R = 6371$ km.
""")
code(r"""
SOURCE = "buzios"          # "buzios", "britain" <<or||ou>> "upload"
UPLOAD_FORMAT = "kml"      # <<only for||só para>> "upload": "kml", "latlon" <<or||ou>> "lonlat"

DATASETS = {"buzios": ("Buzios_Peninsula.kml", "kml"),
            "britain": ("britain_west_gshhg.txt", "latlon")}


def load_text(source, fmt):
    # <<Text of the coastline file: uploaded, local copy, or downloaded from GitHub.||Texto do arquivo da costa: enviado, cópia local ou baixado do GitHub.>>
    if source == "upload":
        from google.colab import files        # <<works only in Colab||só funciona no Colab>>
        name, data = next(iter(files.upload().items()))
        return data.decode("utf-8"), fmt, name
    name, fmt = DATASETS[source]
    try:
        with open(name, encoding="utf-8") as f:
            return f.read(), fmt, name
    except FileNotFoundError:
        return urllib.request.urlopen(REPO + name).read().decode("utf-8"), fmt, name


def parse_coastline(text, fmt):
    # <<(lat, lon) in degrees, from a KML line or a two-column table.||(lat, lon) em graus, de uma linha KML ou de uma tabela com duas colunas.>>
    if fmt == "kml":
        root = ET.fromstring(text)
        lines = [c.text for el in root.iter() if el.tag.endswith(("LineString", "LinearRing"))
                 for c in el.iter() if c.tag.endswith("coordinates")]
        lon, lat = np.array([[float(v) for v in t.split(",")[:2]] for t in lines[0].split()]).T
    else:
        cols = np.loadtxt(io.StringIO(text.replace(",", " ")), comments="#")
        lat, lon = (cols[:, 0], cols[:, 1]) if fmt == "latlon" else (cols[:, 1], cols[:, 0])
    return lat, lon


def to_plane(lat, lon):
    # <<Flat-Earth projection around the mean latitude, in km. Repeated points are removed.||Projeção de Terra plana em torno da latitude média, em km. Pontos repetidos são removidos.>>
    lat0 = np.radians(np.mean(lat))
    x = R_EARTH * np.radians(lon - lon[0]) * np.cos(lat0)
    y = R_EARTH * np.radians(lat - lat[0])
    xy = np.column_stack([x, y])
    keep = np.r_[True, np.any(np.diff(xy, axis=0) != 0.0, axis=1)]
    return xy[keep]


def arc_length(xy):
    # <<Distance along the polyline from its first point to every vertex.||Distância ao longo da poligonal, do primeiro ponto até cada vértice.>>
    return np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]


def points_at(xy, s_knots, s):
    # <<Points of the polyline at distances s along it (any array shape).||Pontos da poligonal nas distâncias s ao longo dela (qualquer formato de array).>>
    return np.stack([np.interp(s, s_knots, xy[:, 0]), np.interp(s, s_knots, xy[:, 1])], axis=-1)


def path_length(P):
    # <<Sum of the straight segments joining consecutive points (last axis = x, y).||Soma dos segmentos retos entre pontos consecutivos (último eixo = x, y).>>
    return np.hypot(*np.moveaxis(np.diff(P, axis=-2), -1, 0)).sum(axis=-1)


text, fmt, name = load_text(SOURCE, UPLOAD_FORMAT)
lat, lon = parse_coastline(text, fmt)
xy = to_plane(lat, lon)
s_knots = arc_length(xy)
L_ref = s_knots[-1]
seg = np.diff(s_knots)

print(f"{name}: {len(xy)} <<points||pontos>>, <<length of the polyline||comprimento da poligonal>> L_ref = {L_ref:.3f} km")
print(f"<<distance between the two ends||distância entre as duas pontas>>: {np.hypot(*(xy[-1] - xy[0])):.2f} km;"
      f" <<median spacing of the points||espaçamento mediano dos pontos>>: {1000 * np.median(seg):.0f} m")

fig, ax = plt.subplots(figsize=(6.5, 6.5))
ax.plot(xy[:, 0], xy[:, 1], color=GREY, lw=1)
ax.plot(*xy[0], "o", color=BLUE, label="<<start||início>>")
ax.plot(*xy[-1], "s", color=ORANGE, label="<<end||fim>>")
ax.set_aspect("equal")
ax.set_xlabel("x (km)")
ax.set_ylabel("y (km)")
ax.set_title(name)
ax.legend()
plt.show()
""")

# =============================================================================== 3
md(r"""
## 3. One measurement with random points

The two ends of the coast are kept, and $n_p - 2$ points are drawn uniformly along its
length. Joining all $n_p$ points in order gives a polyline of $n_p - 1$ straight
segments, whose total length $L$ is the measurement.

**Predict.** Will the measured length $L$ be larger or smaller than $L_\text{ref}$,
the length of the coastline polyline? Can it ever be equal?
""", r"""
## 3. Uma medição com pontos aleatórios

As duas pontas da costa são mantidas, e $n_p - 2$ pontos são sorteados uniformemente
ao longo do seu comprimento. Ligando os $n_p$ pontos em ordem, obtemos uma poligonal
de $n_p - 1$ segmentos retos, cujo comprimento total $L$ é a medição.

**Preveja.** O comprimento medido $L$ será maior ou menor que $L_\text{ref}$, o
comprimento da poligonal da costa? Ele pode ser igual?
""")
code(r"""
def random_points(xy, n_p, n_real, rng):
    # <<n_real measurements with n_p points each: both ends plus n_p - 2 random points.||n_real medições com n_p pontos cada: as duas pontas mais n_p - 2 pontos aleatórios.>>
    s_k = arc_length(xy)
    s = np.sort(rng.random((n_real, n_p - 2)) * s_k[-1], axis=1)
    s = np.column_stack([np.zeros(n_real), s, np.full(n_real, s_k[-1])])
    return points_at(xy, s_k, s)              # <<shape||formato>> (n_real, n_p, 2)


P = random_points(xy, 10, 1, rng)[0]
L = path_length(P)
print(f"n_p = 10:  L = {L:.3f} km,  L / L_ref = {L / L_ref:.3f}")

fig, ax = plt.subplots(figsize=(6.5, 6.5))
ax.plot(xy[:, 0], xy[:, 1], color=GREY, lw=1, label="<<coastline||costa>>")
ax.plot(P[:, 0], P[:, 1], "-o", color=BLUE, ms=5, label=f"<<measurement||medição>>, n_p = 10")
ax.set_aspect("equal")
ax.set_xlabel("x (km)")
ax.set_ylabel("y (km)")
ax.legend()
plt.show()
""")
md(r"""
<details><summary><b>Answer</b></summary>

Always smaller (or equal). Each straight segment is the shortest path between its two
points, so it is never longer than the piece of coastline it replaces (triangle
inequality). Equality needs every piece of coast between two consecutive points to be
straight, which never happens on a real coast.
</details>
""", r"""
<details><summary><b>Resposta</b></summary>

Sempre menor (ou igual). Cada segmento reto é o caminho mais curto entre seus dois
pontos, então nunca é mais longo que o trecho de costa que ele substitui (desigualdade
triangular). A igualdade exigiria que todo trecho de costa entre dois pontos
consecutivos fosse reto, o que nunca acontece numa costa real.
</details>
""")

# =============================================================================== 4
md(r"""
## 4. Many measurements: the sampling distribution

Other random points give another length: $L$ is a random variable. Its distribution
$P(L\,|\,n_p)$ is the **sampling distribution** of the estimator. Its spread is the
variability of the *method*, not an error in the coastline data.

**Predict.** When $n_p$ grows from 10 to 50, does the spread of $L$ grow or shrink?
And the mean?
""", r"""
## 4. Muitas medições: a distribuição amostral

Outros pontos aleatórios dão outro comprimento: $L$ é uma variável aleatória. A sua
distribuição $P(L\,|\,n_p)$ é a **distribuição amostral** do estimador. A sua largura é
a variabilidade do *método*, não um erro nos dados da costa.

**Preveja.** Quando $n_p$ passa de 10 para 50, a largura da distribuição de $L$
aumenta ou diminui? E a média?
""")
code(r"""
N_REAL = 2000                             # <<measurements per n_p||medições por n_p>>

fig, ax = plt.subplots()
for n_p, colour in ((10, BLUE), (50, ORANGE)):
    L = path_length(random_points(xy, n_p, N_REAL, rng))
    skew = np.mean((L - L.mean()) ** 3) / L.std() ** 3
    print(f"n_p = {n_p:3d}:  <<mean||média>> = {L.mean():8.3f} km,  <<std. dev.||desvio-padrão>> = {L.std(ddof=1):6.3f} km,"
          f"  <<relative||relativo>> = {L.std(ddof=1) / L.mean():.3f},  <<skewness||assimetria>> = {skew:+.2f}")
    ax.hist(L, bins=50, density=True, alpha=0.6, color=colour, label=f"n_p = {n_p}")
ax.axvline(L_ref, color=GREY, ls="--", label="L_ref")
ax.set_xlabel("<<measured length||comprimento medido>> L (km)")
ax.set_ylabel("<<probability density||densidade de probabilidade>>")
ax.legend()
plt.show()
""")
md(r"""
<details><summary><b>Answer</b></summary>

The mean grows towards $L_\text{ref}$ (more points follow more of the coast) and the
*relative* spread shrinks. Each measurement averages over more segments.
</details>
""", r"""
<details><summary><b>Resposta</b></summary>

A média cresce em direção a $L_\text{ref}$ (mais pontos acompanham mais detalhes da
costa) e a largura *relativa* diminui. Cada medição faz a média sobre mais segmentos.
</details>
""")

# =============================================================================== 5
md(r"""
## 5. Changing the resolution: the nested sweep

We now measure at many values of $n_p$. In a **nested** sweep each realisation draws
its random points *once*, and the measurement with $n_p$ points uses the first
$n_p - 2$ of them. It is like one surveyor who keeps adding points to the same map:
each $P(L\,|\,n_p)$ is the same as with fresh points, but the values of one
realisation at different $n_p$ are correlated.

**Why a power law?** Richardson found that the length measured with a ruler $\ell$
behaves as $L \propto \ell^{\,1-D}$. With $N = n_p - 1$ segments of typical length
$\ell \approx L/N$, this becomes
$$L = L_0\,(n_p - 1)^{\beta}, \qquad \beta = 1 - \frac{1}{D}, \qquad D = \frac{1}{1-\beta}.$$
A straight line has $D = 1$ ($\beta = 0$: the length does not change); a curve so
wiggly that it fills the plane would have $D = 2$.
""", r"""
## 5. Variando a resolução: a varredura aninhada

Agora medimos em muitos valores de $n_p$. Numa varredura **aninhada** (*nested*), cada
realização sorteia seus pontos aleatórios *uma única vez*, e a medição com $n_p$
pontos usa os $n_p - 2$ primeiros. É como um agrimensor que vai acrescentando pontos ao
mesmo mapa: cada $P(L\,|\,n_p)$ é igual à obtida com pontos novos, mas os valores de
uma mesma realização em diferentes $n_p$ são correlacionados.

**Por que uma lei de potência?** Richardson observou que o comprimento medido com uma
régua $\ell$ se comporta como $L \propto \ell^{\,1-D}$. Com $N = n_p - 1$ segmentos de
tamanho típico $\ell \approx L/N$, isso se torna
$$L = L_0\,(n_p - 1)^{\beta}, \qquad \beta = 1 - \frac{1}{D}, \qquad D = \frac{1}{1-\beta}.$$
Uma reta tem $D = 1$ ($\beta = 0$: o comprimento não muda); uma curva tão sinuosa que
preenchesse o plano teria $D = 2$.
""")
code(r"""
def rp_nested(xy, n_values, n_real, rng):
    # <<Lengths, shape (n_real, len(n_values)). Each n_p uses the first n_p - 2 points of the same draw.||Comprimentos, formato (n_real, len(n_values)). Cada n_p usa os n_p - 2 primeiros pontos do mesmo sorteio.>>
    s_k = arc_length(xy)
    U = rng.random((n_real, max(n_values) - 2)) * s_k[-1]   # <<one draw per realisation||um sorteio por realização>>
    out = np.empty((n_real, len(n_values)))
    for j, n_p in enumerate(n_values):
        s = np.sort(U[:, :n_p - 2], axis=1)
        s = np.column_stack([np.zeros(n_real), s, np.full(n_real, s_k[-1])])
        out[:, j] = path_length(points_at(xy, s_k, s))
    return out


N_VALUES = np.unique(np.rint(np.geomspace(5, 500, 40)).astype(int))
L_rp = rp_nested(xy, N_VALUES, N_REAL, rng)
L_mean = L_rp.mean(axis=0)
lo, hi = np.percentile(L_rp, [16, 84], axis=0)

fig, ax = plt.subplots()
ax.fill_between(N_VALUES - 1, lo, hi, color=BLUE, alpha=0.25, lw=0, label="<<16-84% of the measurements||16-84% das medições>>")
ax.plot(N_VALUES - 1, L_mean, "o-", color=BLUE, ms=3, label="<<mean||média>> <L>")
ax.axhline(L_ref, color=GREY, ls="--", label="L_ref")
ax.set_xscale("log")
ax.set_yscale("log")
plain(ax.yaxis)
ax.set_xlabel("<<number of segments||número de segmentos>> n_p - 1")
ax.set_ylabel("L (km)")
ax.legend()
plt.show()
""")

# =============================================================================== 6
md(r"""
## 6. Fitting the power law, and how uncertain is $D$?

Taking logarithms, $\ln L = \ln L_0 + \beta \ln(n_p - 1)$ is a straight line, fitted by
least squares inside a **window** of $n_p$ values (Section 7 discusses the choice). The
uncertainty of $D$ can be estimated in two ways:

1. **Fit the mean curve** $\langle L\rangle$ once and take the standard error of
   $\beta$ from the usual least-squares formula, then $\sigma_D = D^2\sigma_\beta$
   (since $dD/d\beta = D^2$).
2. **Monte Carlo:** fit every realisation separately and look at the distribution of
   the $D$ values obtained.

**Predict.** Which of the two will give the larger uncertainty?
""", r"""
## 6. Ajustando a lei de potência: qual é a incerteza de $D$?

Tomando logaritmos, $\ln L = \ln L_0 + \beta \ln(n_p - 1)$ é uma reta, ajustada por
mínimos quadrados dentro de uma **janela** de valores de $n_p$ (a escolha é discutida
na Seção 7). A incerteza de $D$ pode ser estimada de duas maneiras:

1. **Ajustar a curva média** $\langle L\rangle$ uma vez e tomar o erro-padrão de
   $\beta$ da fórmula usual de mínimos quadrados; então $\sigma_D = D^2\sigma_\beta$
   (pois $dD/d\beta = D^2$).
2. **Monte Carlo:** ajustar cada realização separadamente e olhar a distribuição dos
   valores de $D$ obtidos.

**Preveja.** Qual das duas vai dar a maior incerteza?
""")
code(r"""
WINDOWS = {"buzios": (10, 50), "britain": (20, 200)}
N_MIN, N_MAX = WINDOWS.get(SOURCE, (10, 50))    # <<fit window in n_p (change it for your own data)||janela de ajuste em n_p (mude para os seus dados)>>


def fit_D(x, L):
    # <<Least squares on ln L = ln L0 + beta ln x for each row of L. Returns beta and D.||Mínimos quadrados em ln L = ln L0 + beta ln x para cada linha de L. Retorna beta e D.>>
    beta = np.polyfit(np.log(x), np.log(np.atleast_2d(L)).T, 1)[0]
    return beta, 1.0 / (1.0 - beta)


win = (N_VALUES >= N_MIN) & (N_VALUES <= N_MAX)
x = N_VALUES[win] - 1.0

# 1. <<mean curve, standard error from the least-squares formula||curva média, erro-padrão da fórmula de mínimos quadrados>>
(beta_m, _), cov = np.polyfit(np.log(x), np.log(L_mean[win]), 1, cov=True)
D_m = 1.0 / (1.0 - beta_m)
sigma_D = D_m ** 2 * np.sqrt(cov[0, 0])

# 2. <<Monte Carlo: one fit per realisation||Monte Carlo: um ajuste por realização>>
_, D_rp = fit_D(x, L_rp[:, win])
q16, q50, q84 = np.percentile(D_rp, [16, 50, 84])

print(f"<<window||janela>>: n_p = {N_MIN}..{N_MAX}  ({win.sum()} <<values||valores>>)")
print(f"1. <<mean curve||curva média>>:  D = {D_m:.4f} ± {sigma_D:.4f}")
print(f"2. Monte Carlo:     D = {q50:.4f}  (16-84%: {q16:.4f} .. {q84:.4f}),  <<half-width||meia-largura>> {0.5 * (q84 - q16):.4f}")

C = np.corrcoef(L_rp[:, win].T)
r = np.mean(np.diag(C, 1))
print(f"<<mean correlation between neighbouring values of n_p in the same realisation||correlação média entre valores vizinhos de n_p na mesma realização>>: {r:.2f}")

fig, ax = plt.subplots()
ax.hist(D_rp, bins=60, density=True, color=BLUE, alpha=0.7, label="Monte Carlo")
ax.axvspan(D_m - sigma_D, D_m + sigma_D, color=ORANGE, alpha=0.5, label="<<mean curve ± standard error||curva média ± erro-padrão>>")
ax.set_xlabel("D")
ax.set_ylabel("<<probability density||densidade de probabilidade>>")
ax.legend()
plt.show()
""")
md(r"""
<details><summary><b>Answer</b></summary>

The Monte-Carlo spread is far larger. The least-squares formula assumes that the points
scatter *independently* around the line. Here the points of the mean curve are averages
of thousands of measurements, with almost no scatter, and neighbouring points of one
realisation are strongly correlated (about 0.9, printed above): when one measurement
happens to fall above the trend, its neighbours in $n_p$ do too. The formula then measures only
how well the mean curve is described by a straight line.

The two numbers answer different questions. The Monte-Carlo spread says how much $D$
would change if the whole measurement were repeated with other random points; the
uncertainty of its median shrinks as $1/\sqrt{N_\text{real}}$ (Exercise 3). Neither
includes the choice of the window, which is the subject of the next section.
</details>
""", r"""
<details><summary><b>Resposta</b></summary>

A largura do Monte Carlo é muito maior. A fórmula de mínimos quadrados supõe que os
pontos se espalham de forma *independente* em torno da reta. Aqui os pontos da curva
média são médias de milhares de medições, quase sem espalhamento, e pontos vizinhos de
uma mesma realização são fortemente correlacionados (cerca de 0,9, impresso acima):
quando uma medição cai acima da tendência, as vizinhas em $n_p$ também caem. A fórmula então
mede apenas o quanto a curva média é bem descrita por uma reta.

Os dois números respondem a perguntas diferentes. A largura do Monte Carlo diz o
quanto $D$ mudaria se a medição inteira fosse repetida com outros pontos aleatórios; a
incerteza da sua mediana diminui como $1/\sqrt{N_\text{real}}$ (Exercício 3). Nenhum
dos dois inclui a escolha da janela, que é o assunto da próxima seção.
</details>
""")

# =============================================================================== 7
md(r"""
## 7. Where should we fit? Local dimension and the scaling window

A power law is a straight line on a log-log plot. Its *local* slope,
$\beta_\text{loc} = d\ln\langle L\rangle / d\ln(n_p - 1)$, gives a local dimension
$D_\text{loc} = 1/(1-\beta_\text{loc})$ at each resolution. It is plotted against the
**effective ruler** $\ell = \langle L\rangle/(n_p - 1)$, the mean length of one segment.

A self-similar curve keeps $D_\text{loc}$ constant over a range of scales: a
**plateau**, where the fit window belongs. At rulers close to the spacing of the data
points the polyline looks straight and $D_\text{loc} \to 1$; at rulers comparable to
the size of the coast the power law cannot hold either.
""", r"""
## 7. Onde ajustar? Dimensão local e a janela de escala

Uma lei de potência é uma reta num gráfico log-log. A sua inclinação *local*,
$\beta_\text{loc} = d\ln\langle L\rangle / d\ln(n_p - 1)$, dá uma dimensão local
$D_\text{loc} = 1/(1-\beta_\text{loc})$ em cada resolução. Ela é desenhada em função
da **régua efetiva** $\ell = \langle L\rangle/(n_p - 1)$, o comprimento médio de um
segmento.

Uma curva autossimilar mantém $D_\text{loc}$ constante numa faixa de escalas: um
**plateau**, onde a janela de ajuste deve ficar. Para réguas próximas do espaçamento
dos pontos dos dados, a poligonal parece reta e $D_\text{loc} \to 1$; para réguas
comparáveis ao tamanho da costa, a lei de potência também não vale.
""")
code(r"""
def local_dimension(n_values, L_mean):
    beta = np.gradient(np.log(L_mean), np.log(np.asarray(n_values) - 1.0))
    return 1.0 / (1.0 - beta)


ell = L_mean / (N_VALUES - 1)                    # <<effective ruler||régua efetiva>>, km
D_loc = local_dimension(N_VALUES, L_mean)

fig, ax = plt.subplots()
ax.plot(ell, D_loc, "o-", color=BLUE, ms=3)
ax.axvspan(ell[win].min(), ell[win].max(), color=BLUE, alpha=0.12, label=f"<<fit window||janela de ajuste>> n_p = {N_MIN}..{N_MAX}")
ax.axhline(1.0, color=GREY, ls=":")
ax.set_xscale("log")
ax.set_xlabel("<<effective ruler||régua efetiva>> <L>/(n_p - 1)  (km)")
ax.set_ylabel("<<local dimension||dimensão local>> D_loc")
ax.legend()
plt.show()

print("<<window||janela>> n_p       <<median||mediana>> D   16-84%")
for a, b in ((5, 20), (10, 50), (20, 100), (50, 200), (100, 500)):
    w = (N_VALUES >= a) & (N_VALUES <= b)
    _, D = fit_D(N_VALUES[w] - 1.0, L_rp[:, w])
    print(f"{a:4d} .. {b:<4d}      {np.median(D):.4f}     {np.percentile(D, 16):.4f} .. {np.percentile(D, 84):.4f}")
""")
md(r"""
The spread inside one window is the **statistical** uncertainty; the change of the
median from one reasonable window to another is a **systematic** uncertainty, and it
is often the larger of the two. A result should state both, and show the plateau that
justifies the window.
""", r"""
A largura dentro de uma janela é a incerteza **estatística**; a mudança da mediana de
uma janela razoável para outra é uma incerteza **sistemática**, muitas vezes a maior
das duas. Um resultado deve informar as duas e mostrar o plateau que justifica a
janela.
""")

# =============================================================================== 8
md(r"""
## 8. The yardstick

A survey starts at a random point of the coast and walks in both directions. From each
station, the next one is the **first** point further along the coastline whose
straight-line distance is exactly $d$. On a polyline this point is found exactly:

1. go through the vertices ahead and stop at the first one, $\mathbf V$, that lies
   outside the circle of radius $d$ around the station $\mathbf P$;
2. the crossing lies on the segment that ends at $\mathbf V$, from its inner end
   $\mathbf E$ (inside the circle). Writing $\mathbf X = \mathbf E + w(\mathbf V-\mathbf E)$
   with $0 < w \le 1$, the condition $|\mathbf X - \mathbf P| = d$ is a quadratic
   equation,
   $$a w^2 + 2 b w + c = 0,\quad a = |\mathbf V-\mathbf E|^2,\quad
   b = (\mathbf E-\mathbf P)\cdot(\mathbf V-\mathbf E),\quad c = |\mathbf E-\mathbf P|^2 - d^2,$$
   whose root is $w = \left(-b + \sqrt{b^2 - ac}\right)/a$.

At each end of the line a piece shorter than $d$ remains. If it is shorter than $d/2$,
the last station is removed and one segment runs straight to the end; otherwise the
piece is kept as one more, shorter, segment. Either way $n_p - 1 \approx L/d$, and
$n_p$ counts all stations, the two ends included, as for random points.

All rulers use the same random starting points (*common seeds*), so each survey is one
realisation across all rulers. The rulers are chosen to match the RP window: from the
effective ruler at $n_p = $ `N_MAX` to the one at $n_p = $ `N_MIN`.
""", r"""
## 8. A régua

Um levantamento começa num ponto aleatório da costa e caminha nos dois sentidos. A
partir de cada estação, a próxima é o **primeiro** ponto mais adiante na costa cuja
distância em linha reta é exatamente $d$. Numa poligonal, esse ponto é encontrado de
forma exata:

1. percorremos os vértices à frente e paramos no primeiro, $\mathbf V$, que fica fora
   do círculo de raio $d$ em torno da estação $\mathbf P$;
2. o cruzamento está no segmento que termina em $\mathbf V$, a partir da sua
   extremidade interna $\mathbf E$ (dentro do círculo). Escrevendo
   $\mathbf X = \mathbf E + w(\mathbf V-\mathbf E)$ com $0 < w \le 1$, a condição
   $|\mathbf X - \mathbf P| = d$ é uma equação do segundo grau,
   $$a w^2 + 2 b w + c = 0,\quad a = |\mathbf V-\mathbf E|^2,\quad
   b = (\mathbf E-\mathbf P)\cdot(\mathbf V-\mathbf E),\quad c = |\mathbf E-\mathbf P|^2 - d^2,$$
   cuja raiz é $w = \left(-b + \sqrt{b^2 - ac}\right)/a$.

Em cada ponta da linha sobra um pedaço menor que $d$. Se ele for menor que $d/2$, a
última estação é removida e um segmento vai direto até a ponta; senão, o pedaço é
mantido como mais um segmento, mais curto. Nos dois casos $n_p - 1 \approx L/d$, e
$n_p$ conta todas as estações, incluindo as duas pontas, como nos pontos aleatórios.

Todas as réguas usam os mesmos pontos de partida aleatórios (*sementes comuns*), então
cada levantamento é uma realização em todas as réguas. As réguas são escolhidas para
corresponder à janela do RP: da régua efetiva em $n_p = $ `N_MAX` até a de
$n_p = $ `N_MIN`.
""")
code(r"""
def walk(xy, s_k, s0, d):
    # <<Stations of a walk with ruler d, from distance s0 along the line to its end.||Estações de uma caminhada com régua d, da distância s0 ao longo da linha até o fim.>>
    i = min(np.searchsorted(s_k, s0, side="right") - 1, len(xy) - 2)   # <<segment of the start||segmento da partida>>
    P = points_at(xy, s_k, s0)
    stations = [P]
    while True:
        # 1. <<first vertex ahead outside the circle of radius d (searched in blocks of 64)||primeiro vértice à frente fora do círculo de raio d (busca em blocos de 64)>>
        k, start = None, i + 1
        while start < len(xy):
            stop = min(start + 64, len(xy))
            outside = np.flatnonzero(np.hypot(*(xy[start:stop] - P).T) >= d)
            if outside.size:
                k = start + outside[0]
                break
            start = stop
        if k is None:                         # <<the end of the line is closer than d||o fim da linha está a menos de d>>
            return np.array(stations)
        # 2. <<crossing on the segment that ends at vertex k||cruzamento no segmento que termina no vértice k>>
        E = P if k - 1 == i else xy[k - 1]
        V = xy[k]
        a = (V - E) @ (V - E)
        b = (E - P) @ (V - E)
        c = (E - P) @ (E - P) - d * d
        P = E + (-b + np.sqrt(b * b - a * c)) / a * (V - E)
        stations.append(P)
        i = k - 1


def yardstick(xy, d, s0):
    # <<One survey from the seed s0, walking both ways. Returns (L, n_p, stations).||Um levantamento a partir da semente s0, nos dois sentidos. Retorna (L, n_p, estações).>>
    s_k = arc_length(xy)
    rev = xy[::-1]
    forward = walk(xy, s_k, s0, d)
    backward = walk(rev, arc_length(rev), s_k[-1] - s0, d)
    # <<end piece shorter than d/2: drop the last station, one segment runs to the end||pedaço da ponta menor que d/2: remove a última estação, um segmento vai até a ponta>>
    if len(forward) > 1 and np.hypot(*(xy[-1] - forward[-1])) < d / 2:
        forward = forward[:-1]
    if len(backward) > 1 and np.hypot(*(xy[0] - backward[-1])) < d / 2:
        backward = backward[:-1]
    path = np.vstack([xy[:1], backward[::-1], forward[1:], xy[-1:]])
    return path_length(path), len(path), path


ell_win = ell[win]
RULERS = np.geomspace(ell_win.min(), ell_win.max(), 12)
N_SURVEYS = 300 if len(xy) < 5000 else 100     # <<fewer surveys for long coastlines||menos levantamentos para costas longas>>

# <<one survey, drawn on the map||um levantamento, desenhado no mapa>>
L1, n1, path = yardstick(xy, RULERS[-1], 0.5 * L_ref)
fig, ax = plt.subplots(figsize=(6.5, 6.5))
ax.plot(xy[:, 0], xy[:, 1], color=GREY, lw=1, label="<<coastline||costa>>")
ax.plot(path[:, 0], path[:, 1], "-o", color=ORANGE, ms=4, label=f"<<yardstick||régua>> d = {RULERS[-1]:.3g} km: n_p = {n1}, L = {L1:.2f} km")
ax.set_aspect("equal")
ax.set_xlabel("x (km)")
ax.set_ylabel("y (km)")
ax.legend()
plt.show()
""")
code(r"""
s0 = rng.random(N_SURVEYS) * L_ref               # <<common seeds for every ruler||sementes comuns para todas as réguas>>
L_ys = np.empty((N_SURVEYS, RULERS.size))
n_ys = np.empty((N_SURVEYS, RULERS.size), dtype=int)
for j, d in enumerate(RULERS):
    for k in range(N_SURVEYS):
        L_ys[k, j], n_ys[k, j], _ = yardstick(xy, d, s0[k])

# <<Richardson: ln L = const + (1 - D) ln d, one fit per survey||Richardson: ln L = const + (1 - D) ln d, um ajuste por levantamento>>
slope = np.polyfit(np.log(RULERS), np.log(L_ys).T, 1)[0]
D_ys_d = 1.0 - slope
# <<as for random points: ln L against ln(n_p - 1), one fit per survey||como nos pontos aleatórios: ln L contra ln(n_p - 1), um ajuste por levantamento>>
D_ys_n = np.array([fit_D(n_ys[k] - 1.0, L_ys[k])[1][0] for k in range(N_SURVEYS)])

for label, D in (("<<L against d||L contra d>>      ", D_ys_d), ("<<L against n_p - 1||L contra n_p - 1>>", D_ys_n)):
    print(f"YS, {label}:  D = {np.median(D):.4f}  (16-84%: {np.percentile(D, 16):.4f} .. {np.percentile(D, 84):.4f})")

fig, ax = plt.subplots()
ax.fill_between(RULERS, *np.percentile(L_ys, [16, 84], axis=0), color=ORANGE, alpha=0.25, lw=0, label="16-84%")
ax.plot(RULERS, L_ys.mean(axis=0), "s-", color=ORANGE, ms=4, label="<<mean||média>> <L>")
ax.set_xscale("log")
ax.set_yscale("log")
plain(ax.xaxis)
plain(ax.yaxis)
ax.set_xlabel("<<ruler||régua>> d (km)")
ax.set_ylabel("L (km)")
ax.set_title("<<Richardson plot||Gráfico de Richardson>>")
ax.legend()
plt.show()
""")

# =============================================================================== 9
md(r"""
## 9. Two estimators, one curve

The two estimators are built differently: random points fix the *number* of segments
and let their lengths vary; the yardstick fixes the *length* of the segments and lets
their number vary. Plotted against the length of one segment, the effective ruler
$\ell$ for random points and $d$ for the yardstick, both describe the same coastline.

**Predict.** Will the two curves coincide?
""", r"""
## 9. Dois estimadores, uma curva

Os dois estimadores são construídos de forma diferente: os pontos aleatórios fixam o
*número* de segmentos e deixam seus comprimentos variar; a régua fixa o *comprimento*
dos segmentos e deixa o seu número variar. Desenhados em função do comprimento de um
segmento, a régua efetiva $\ell$ nos pontos aleatórios e $d$ na régua, os dois
descrevem a mesma costa.

**Preveja.** As duas curvas vão coincidir?
""")
code(r"""
fig, ax = plt.subplots()
ax.plot(ell, L_mean, "o-", color=BLUE, ms=3, label="<<random points||pontos aleatórios>>: <L> <<against||contra>> <L>/(n_p - 1)")
ax.plot(RULERS, L_ys.mean(axis=0), "s-", color=ORANGE, ms=4, label="<<yardstick||régua>>: <L> <<against||contra>> d")
ax.axvspan(ell_win.min(), ell_win.max(), color=GREY, alpha=0.1)
ax.set_xscale("log")
ax.set_yscale("log")
plain(ax.yaxis)
ax.set_xlabel("<<length of one segment||comprimento de um segmento>> (km)")
ax.set_ylabel("<L> (km)")
ax.legend()
plt.show()

print(f"<<random points||pontos aleatórios>>, n_p = {N_MIN}..{N_MAX}:   D = {np.median(D_rp):.4f}  (16-84%: {np.percentile(D_rp, 16):.4f} .. {np.percentile(D_rp, 84):.4f})")
print(f"<<yardstick||régua>>, d = {RULERS[0]:.3g}..{RULERS[-1]:.3g} km:  D = {np.median(D_ys_d):.4f}  (16-84%: {np.percentile(D_ys_d, 16):.4f} .. {np.percentile(D_ys_d, 84):.4f})")
""")
md(r"""
<details><summary><b>Answer</b></summary>

Nearly, but not exactly: a random-points segment and a ruler of the same mean length
do not cut the same corners, because random segments vary in length. What matters is
that both give a similar *slope*, hence a similar $D$. The yardstick usually gives a
narrower distribution of $D$: its segments all have the same length, so each survey
has less randomness.
</details>
""", r"""
<details><summary><b>Resposta</b></summary>

Quase, mas não exatamente: um segmento dos pontos aleatórios e uma régua com o mesmo
comprimento médio não cortam as mesmas curvas, porque os segmentos aleatórios variam
de tamanho. O que importa é que os dois dão uma *inclinação* parecida, e portanto um
$D$ parecido. A régua costuma dar uma distribuição de $D$ mais estreita: todos os seus
segmentos têm o mesmo comprimento, então cada levantamento tem menos aleatoriedade.
</details>
""")

# =============================================================================== 10
md(r"""
## 10. Validation: does the method find a dimension we know?

Before trusting a measurement, test the method where the answer is known. Two
synthetic curves of the same size as the Búzios coast (7 km between the ends, about
38–39 km long):

- a **Koch curve**: each segment is replaced by four segments of a third of its length,
  six times over, so $D = \ln 4 / \ln 3 \approx 1.262$ exactly;
- a **smooth sinusoid**, three periods: a smooth curve, so $D = 1$.

Both are measured with random points and fitted in the same window $n_p$ = 10..50.

**Predict.** What will the fit give for the sinusoid?
""", r"""
## 10. Validação: o método encontra uma dimensão conhecida?

Antes de confiar numa medição, teste o método onde a resposta é conhecida. Duas curvas
sintéticas do mesmo tamanho da costa de Búzios (7 km entre as pontas, cerca de
38–39 km de comprimento):

- uma **curva de Koch**: cada segmento é substituído por quatro segmentos de um terço
  do seu comprimento, seis vezes seguidas, então $D = \ln 4 / \ln 3 \approx 1{,}262$
  exatamente;
- uma **senoide suave**, com três períodos: uma curva lisa, então $D = 1$.

As duas são medidas com pontos aleatórios e ajustadas na mesma janela $n_p$ = 10..50.

**Preveja.** O que o ajuste vai dar para a senoide?
""")
code(r"""
def koch(level, base=7.0):
    # <<Koch curve from (0, 0) to (base, 0): each segment becomes four of a third of its length.||Curva de Koch de (0, 0) a (base, 0): cada segmento vira quatro com um terço do seu comprimento.>>
    P = np.array([[0.0, 0.0], [base, 0.0]])
    turn = np.array([[0.5, -np.sqrt(3) / 2], [np.sqrt(3) / 2, 0.5]])   # <<rotation by 60°||rotação de 60°>>
    for _ in range(level):
        A, B = P[:-1], P[1:]
        third = (B - A) / 3
        P = np.vstack([np.stack([A, A + third, A + third + third @ turn.T, A + 2 * third], axis=1).reshape(-1, 2), P[-1:]])
    return P


t = np.linspace(0.0, 1.0, 4001)
TEST_CURVES = {
    "Koch (D = 1.262)": koch(6),
    "<<sinusoid||senoide>> (D = 1)": np.column_stack([7.0 * t, 3.1 * np.sin(6 * np.pi * t)]),
}
w_test = (N_VALUES >= 10) & (N_VALUES <= 50)

fig, ax = plt.subplots()
for (label, curve), colour in zip(TEST_CURVES.items(), (ORANGE, GREEN)):
    Lc = rp_nested(curve, N_VALUES, N_REAL, rng)
    _, Dc = fit_D(N_VALUES[w_test] - 1.0, Lc[:, w_test])
    mean_c = Lc.mean(axis=0)
    print(f"{label:22s} <<length||comprimento>> {arc_length(curve)[-1]:5.1f} km:  <<fitted||ajustado>> D = {np.median(Dc):.3f}  (16-84%: {np.percentile(Dc, 16):.3f} .. {np.percentile(Dc, 84):.3f})")
    ax.plot(mean_c / (N_VALUES - 1), local_dimension(N_VALUES, mean_c), "o-", ms=3, color=colour, label=label)
ax.plot(ell, D_loc, "o-", ms=3, color=BLUE, label=f"{name}")
ax.axhline(np.log(4) / np.log(3), color=ORANGE, ls=":")
ax.axhline(1.0, color=GREEN, ls=":")
ax.set_xscale("log")
ax.set_ylim(0.95, 1.8)
ax.set_xlabel("<<effective ruler||régua efetiva>> <L>/(n_p - 1)  (km)")
ax.set_ylabel("<<local dimension||dimensão local>> D_loc")
ax.legend()
plt.show()
""")
md(r"""
<details><summary><b>Answer</b></summary>

The Koch curve comes out close to 1.262, a little below it: at rulers this coarse,
random points underestimate by a few hundredths, and the local-dimension plot shows the
Koch plateau at 1.26 at smaller rulers. The smooth sinusoid, whose true
dimension is 1, gives a value well **above** 1 in this window, because its wiggles are
about the size of the window. A fitted $D > 1$ in one window is therefore *not*, on its
own, evidence of fractal structure. The local-dimension plot tells them apart: the Koch
curve has a long plateau, while the sinusoid tends to 1 at small rulers. This is why the
plateau must always be shown with the result.
</details>
""", r"""
<details><summary><b>Resposta</b></summary>

A curva de Koch dá um valor próximo de 1,262, um pouco abaixo: com réguas tão
grandes, os pontos aleatórios subestimam em alguns centésimos, e o gráfico da dimensão
local mostra o plateau do Koch em 1,26 nas réguas menores. A senoide suave, cuja
dimensão verdadeira é 1, dá um valor bem **acima** de 1 nesta janela, porque as suas
ondulações têm mais ou menos o tamanho da janela. Um $D > 1$ ajustado numa janela,
sozinho, *não* é evidência de estrutura fractal. O gráfico da dimensão local separa os
dois casos: a curva de Koch tem um plateau longo, enquanto a senoide tende a 1 nas
réguas pequenas. Por isso o plateau deve sempre acompanhar o resultado.
</details>
""")

# =============================================================================== 11
md(r"""
## 11. Your turn

1. **Your own coast.** Draw a coastline in Google My Maps, export it as KML, set
   `SOURCE = "upload"` and run everything again. Choose `N_MIN` and `N_MAX` from the
   plateau of Section 7.
2. **Is the Earth flat enough?** Write the great-circle (haversine) distance
   $$\ell = 2R\arcsin\sqrt{\sin^2\tfrac{\Delta\varphi}{2} + \cos\varphi_1\cos\varphi_2\sin^2\tfrac{\Delta\lambda}{2}},$$
   use it in place of `path_length` for `SOURCE = "britain"` (950 km from south to
   north), and compare $L$ and $D$ with the flat-Earth values.
3. **More realisations.** Double `N_REAL`. Which changes more: the median of $D$ or the
   width of its distribution? Why?
4. **Systematic uncertainty.** Find the widest window in which the median of $D$
   changes by less than 0.01. Report $D$ with both its statistical and systematic
   uncertainties.
5. **Independent instead of nested.** Change `rp_nested` so that each $n_p$ draws new
   points. Does the distribution of $D$ become wider or narrower? Explain with the
   correlation of Section 6.
6. **The yardstick on the Koch curve.** Run Section 8 with `xy = koch(6)`. How does
   $n_p$ change with $d$? (Look at its values: the Koch curve is self-similar with a
   factor of 3.)
""", r"""
## 11. Sua vez

1. **A sua costa.** Desenhe uma linha costeira no Google My Maps, exporte como KML,
   coloque `SOURCE = "upload"` e rode tudo de novo. Escolha `N_MIN` e `N_MAX` a partir
   do plateau da Seção 7.
2. **A Terra é plana o suficiente?** Escreva a distância de grande círculo (haversine)
   $$\ell = 2R\arcsin\sqrt{\sin^2\tfrac{\Delta\varphi}{2} + \cos\varphi_1\cos\varphi_2\sin^2\tfrac{\Delta\lambda}{2}},$$
   use-a no lugar de `path_length` com `SOURCE = "britain"` (950 km de sul a norte) e
   compare $L$ e $D$ com os valores da Terra plana.
3. **Mais realizações.** Dobre `N_REAL`. O que muda mais: a mediana de $D$ ou a
   largura da sua distribuição? Por quê?
4. **Incerteza sistemática.** Encontre a janela mais larga em que a mediana de $D$
   muda menos de 0,01. Informe $D$ com as incertezas estatística e sistemática.
5. **Independente em vez de aninhado.** Mude `rp_nested` para que cada $n_p$ sorteie
   pontos novos. A distribuição de $D$ fica mais larga ou mais estreita? Explique com a
   correlação da Seção 6.
6. **A régua na curva de Koch.** Rode a Seção 8 com `xy = koch(6)`. Como $n_p$ muda
   com $d$? (Olhe os valores: a curva de Koch é autossimilar com fator 3.)
""")

# =============================================================================== 12
md(rf"""
## 12. Going further

- The full analysis code (on the sphere, with spline interpolation, closed islands,
  10⁵–10⁶ realisations and several fitting options) and the example datasets are in
  the repository [{REPO}](https://github.com/{REPO}).
- L. F. Richardson, "The problem of contiguity: an appendix to *Statistics of deadly
  quarrels*", *General Systems Yearbook* **6**, 139–187 (1961).
- B. Mandelbrot, "How long is the coast of Britain? Statistical self-similarity and
  fractional dimension", *Science* **156**, 636–638 (1967).
- P. Wessel and W. H. F. Smith, "A global, self-consistent, hierarchical,
  high-resolution shoreline database", *J. Geophys. Res.* **101**(B4), 8741–8743
  (1996): the source of the Britain data (GSHHG).
""", rf"""
## 12. Para ir além

- O código completo da análise (na esfera, com interpolação por splines, ilhas
  fechadas, 10⁵–10⁶ realizações e várias opções de ajuste) e os conjuntos de dados de
  exemplo estão no repositório [{REPO}](https://github.com/{REPO}).
- L. F. Richardson, "The problem of contiguity: an appendix to *Statistics of deadly
  quarrels*", *General Systems Yearbook* **6**, 139–187 (1961).
- B. Mandelbrot, "How long is the coast of Britain? Statistical self-similarity and
  fractional dimension", *Science* **156**, 636–638 (1967).
- P. Wessel e W. H. F. Smith, "A global, self-consistent, hierarchical,
  high-resolution shoreline database", *J. Geophys. Res.* **101**(B4), 8741–8743
  (1996): a fonte dos dados da Grã-Bretanha (GSHHG).
""")


# ================================================================================ build
_PAIR = re.compile(r"<<(.*?)\|\|(.*?)>>")


def build(lang: str) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {"name": "python"},
        "colab": {"provenance": []},
    }
    for cell in CELLS:
        if cell[0] == "md":
            nb.cells.append(nbf.v4.new_markdown_cell(cell[1] if lang == "en" else cell[2]))
        else:
            src = _PAIR.sub(lambda m: m.group(1) if lang == "en" else m.group(2), cell[1])
            nb.cells.append(nbf.v4.new_code_cell(src))
    return nb


def main() -> None:
    for lang in ("en", "pt"):
        path = HERE / f"coastline_notebook_{lang}.ipynb"
        nbf.write(build(lang), path)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
