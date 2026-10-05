import marimo

__generated_with = "0.13.4"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Worst-case risk analysis""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Covariance uncertainty

    In this example we do worst-case risk analysis using CVXPY. 
    Our setting is a single period Markowitz portfolio allocation problem.
    We have a fixed portfolio allocation $w \in {\bf R}^n$. The return covariance $\Sigma$ is not known,
    but we believe $\Sigma \in \mathcal S$.
    Here $\mathcal S$ is a convex set of possible covariance matrices.
    The risk is $w^T \Sigma w$, a linear function of $\Sigma$.

    We can compute the worst (maximum) risk, over all possible covariance matrices by solving the convex optimization 
    problem

    $$
    \begin{array}{ll} \text{maximize} & w^T\Sigma w \\
    \text{subject to} & \Sigma \in \mathcal S, \quad \Sigma \succeq 0,
    \end{array}
    $$

    with variable $\Sigma$.

    If the worst-case risk is not too bad, you can worry less.
    If not, you'll confront your worst nightmare.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Example

    In the following code we solve the portfolio allocation problem

    $$
    \begin{array}{ll} \text{minimize} & w^T\Sigma_\mathrm{nom} w \\
    \text{subject to} & {\bf 1}^Tw = 1, \quad \mu^Tw \geq 0.1, \quad \|w\|_1 \leq 2, 
    \end{array}
    $$

    and then compute the worst-case risk under the assumption that $\mathcal S = \left\{ \Sigma^\mathrm{nom} + \Delta \,:\,
    |\Delta_{ii}| =0, \;
    |\Delta_{ij}| \leq 0.2
    \right\}$.

    We might expect that $|\Delta_{ij}| = 0.2$ for all $i \neq j$. 
    This does not happen however because of the constraint that $\Sigma^\mathrm{nom} + \Delta$ is positive semidefinite.
    """
    )
    return


@app.cell
def _():
    import numpy as np

    np.random.seed(2)
    n = 5
    mu = np.abs(np.random.randn(n)) / 15
    _Sigma = np.random.uniform(-0.15, 0.8, size=(n, n))
    Sigma_nom = _Sigma.T @ _Sigma
    print('Sigma_nom =')
    print(np.round(Sigma_nom, decimals=2))
    return Sigma_nom, mu, n, np


@app.cell
def _(Sigma_nom, mu, n, np):
    import cvxpy as cp

    w = cp.Variable(n)
    ret = mu @ w
    _risk = cp.quad_form(w, Sigma_nom)
    _prob = cp.Problem(cp.Minimize(_risk), [cp.sum(w) == 1, ret >= 0.1, cp.norm(w, 1) <= 2])
    _prob.solve()
    print('w =')
    print(np.round(w.value, decimals=2))
    return cp, w


@app.cell
def _(Sigma_nom, cp, n, np, w):
    _Sigma = cp.Variable((n, n), PSD=True)
    Delta = cp.Variable((n, n), symmetric=True)
    _risk = cp.quad_form(w.value, _Sigma)
    _prob = cp.Problem(cp.Maximize(_risk), [_Sigma == Sigma_nom + Delta, cp.diag(Delta) == 0, cp.abs(Delta) <= 0.2])
    _prob.solve()  # does not work in WebAssembly, but when run on server or locally
    print('standard deviation =', np.sqrt(w.value @ Sigma_nom @ w.value))
    print('worst-case standard deviation =', np.sqrt(_risk.value))
    print('worst-case Delta =')
    print(np.round(Delta.value, decimals=2))
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
