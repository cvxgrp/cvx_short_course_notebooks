import marimo

__generated_with = "0.13.4"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Optimal advertising""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Ad display

    In this example we solve a simple optimal advertising problem in CVXPY.
    We have $m$ advertisers/ads, $i=1, \ldots, m$, and $n$ time slots, $t=1, \ldots, n$.
    $T_t$ is the total traffic in time slot $t$.
    $D_{it} \geq 0$ is the number of ad $i$ displayed in period $t$.

    Our constraints are that $\sum_i D_{it} \leq T_t$ and that we satisfy our contracts for minimum total displays: $\sum_t D_{it} \geq c_i$.
    Our goal is to choose $D_{it}$.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Clicks and revenue

    $C_{it}$ is the number of clicks on ad $i$ in period $t$.
    Our click model is $C_{it} = P_{it}D_{it}$, where $P_{it} \in [0,1]$ is the fraction of displays of ad $i$ in time $t$ that translate into clicks.

    We are paid $R_i>0$ per click for ad $i$, up to a budget $B_i$.
    Our total ad revenue is thus

    $$
    S_i = \min \left\{ R_i \sum_t C_{it}, B_i\right\},
    $$

    a concave function of $D$.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Ad optimization
    We choose the displays to maximize revenue, i.e., we solve the optimization problem

    $$
    \begin{array}{ll} \text{maximize} & \sum_i S_i \\
    \text{subject to} & D \geq 0, \quad D^T{\bf 1} \leq T, \quad
    D {\bf 1} \geq c
    \end{array}
    $$

    where the optimization variable is $D\in {\bf R}^{m \times n}$ and $T \in {\bf R}^n$, $c \in {\bf R}^m$, $R \in {\bf R}^m$, $B \in {\bf R}^m$, and $P \in {\bf R}^{m \times n}$ are problem data.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Example

    In the following code we generate and solve an ad optimization problem with 24 hourly periods and 5 ads (A-E).
    """
    )
    return


@app.cell
def _():
    # Generate data for optimal advertising problem.
    import numpy as np

    np.random.seed(1)
    m = 5
    n = 24
    SCALE = 10000
    B = np.random.lognormal(mean=8, size=m) + 10000
    B = 1000 * np.round(B / 1000)

    P_ad = np.random.uniform(size=m)
    P_time = np.random.uniform(size=n)
    P = np.outer(P_ad, P_time)

    T = np.sin(np.linspace(-np.pi, np.pi, n)) * SCALE
    T += -np.min(T) + SCALE
    c = np.random.uniform(size=m)
    c *= 0.6 * T.sum() / c.sum()
    c = 1000 * np.round(c / 1000)
    R = np.random.lognormal(c.min() / c)
    return B, P, R, T, c, m, n, np


@app.cell
def _(B, P, R, T, c, m, n):
    # Form and solve the optimal advertising problem.
    import cvxpy as cp

    D = cp.Variable((m, n))
    S = cp.minimum(cp.multiply(R, cp.sum(cp.multiply(P, D), axis=1)), B)
    prob = cp.Problem(
        cp.Maximize(cp.sum(S)), [D >= 0, cp.sum(D, axis=0) <= T, cp.sum(D, axis=1) >= c]
    )
    prob.solve()
    return (D,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We plot the total traffic $T$ below.""")
    return


@app.cell
def _(T):
    # Plot traffic.
    import matplotlib.pyplot as plt

    plt.plot(T)
    plt.xlabel("Hour")
    plt.ylabel("Traffic")
    plt.show()
    return (plt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We plot the conversion fractions $P$ below.""")
    return


@app.cell
def _(P, n, np, plt):
    _column_labels = range(n)
    _row_labels = list('ABCDE')
    _fig, _ax = plt.subplots()
    _ax.pcolor(P, cmap=plt.cm.Blues)
    _ax.set_xticks(np.arange(P.shape[1]) + 0.5, minor=False)
    _ax.set_yticks(np.arange(P.shape[0]) + 0.5, minor=False)
    _ax.invert_yaxis()
    _ax.xaxis.tick_top()
    _ax.set_xticklabels(_column_labels, minor=False)
    _ax.set_yticklabels(_row_labels, minor=False)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We plot the optimal displays $D$ below.""")
    return


@app.cell
def _(D, n, np, plt):
    _column_labels = range(n)
    _row_labels = list('ABCDE')
    _fig, _ax = plt.subplots()
    _data = D.value
    _ax.pcolor(_data, cmap=plt.cm.Blues)
    _ax.set_xticks(np.arange(_data.shape[1]) + 0.5, minor=False)
    _ax.set_yticks(np.arange(_data.shape[0]) + 0.5, minor=False)
    _ax.invert_yaxis()
    _ax.xaxis.tick_top()
    _ax.set_xticklabels(_column_labels, minor=False)
    _ax.set_yticklabels(_row_labels, minor=False)
    plt.show()
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
