import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Simple portfolio optimization
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We consider a portfolio optimization problem as described on the *Convex Optimization Applications* slides.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## (a)
    Find minimum-risk portfolios with the same expected return as the uniform portfolio ($w =
    (1/n)1$), with risk measured by portfolio return variance, and the following portfolio constraints
    (in addition to $1^Tw = 1$):

    * No (additional) constraints.
    * Long-only: $w \geq 0$.
    * Limit on total short position: $1^T(w_-) \leq 0.5$, where $(w_-)_i = \max\{−w_i, 0\}$.

    Compare the optimal risk in these portfolios with each other and the uniform portfolio.
    """)
    return


@app.cell
def _():
    # Construct problem data.
    import numpy as np

    np.random.seed(1)
    n = 20
    mu = np.ones(n) * 0.03 + np.random.rand(n) * 0.12
    mu[0] = 0
    S = np.random.randn(n, n)
    S = S.T @ S
    Sigma = S / np.max(np.diag(S)) * 0.2
    Sigma[:, 0] = 0
    Sigma[0, :] = 0
    w_unif = np.ones(n) / n
    return Sigma, n, np, w_unif


@app.cell
def _(Sigma, n, np, w_unif):
    import cvxpy as cp
    _w = cp.Variable(n)
    print(f'Risk for uniform: {np.sqrt(w_unif @ Sigma @ w_unif):.1%}')

    # TODO: your code here
    # Solve the problem with no additional constraints; store the
    # portfolio return variance (a CVXPY expression) in _risk.
    print(f'Risk for unconstrained: {np.sqrt(_risk.value):.1%}')

    # TODO: your code here
    # Solve the long-only problem; store the variance in _risk.
    print(f'Risk for long only: {np.sqrt(_risk.value):.1%}')

    # TODO: your code here
    # Solve the problem with the limit on total short position; store
    # the variance in _risk.
    print(f'Risk for limit on short: {np.sqrt(_risk.value):.1%}')
    return (cp,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## (b)

    Plot the optimal risk-return trade-off curves for the long-only portfolio, and for total short position limited to 0.5, in the same figure.

    Comment on the relationship between the two trade-off curves.
    """)
    return


@app.cell
def _(cp, n, np):
    import matplotlib.pyplot as plt
    _w = cp.Variable(n)
    gamma = cp.Parameter(nonneg=True)
    N = 128
    gamma_vals = np.logspace(-1, 5, num=N)
    return N, gamma, gamma_vals, plt


@app.cell
def _(N, gamma, gamma_vals, np, plt):
    # TODO: your code here
    # Define the long-only problem _prob, with expected return
    # _expec_return and variance _risk traded off by gamma.
    return_vec1 = np.zeros(N)
    risk_vec1 = np.zeros(N)
    for _i in range(N):
        gamma.value = gamma_vals[_i]
        _prob.solve()
        return_vec1[_i] = _expec_return.value
        risk_vec1[_i] = _risk.value
    plt.figure()
    plt.plot(np.sqrt(risk_vec1) * 100, return_vec1 * 100, label='Long only')
    return


@app.cell
def _(N, gamma, gamma_vals, np, plt):
    # TODO: your code here
    # Redefine _prob for total short position limited to 0.5.
    return_vec2 = np.zeros(N)
    risk_vec2 = np.zeros(N)
    for _i in range(N):
        gamma.value = gamma_vals[_i]
        _prob.solve()
        return_vec2[_i] = _expec_return.value
        risk_vec2[_i] = _risk.value
    plt.plot(np.sqrt(risk_vec2) * 100, return_vec2 * 100, label='Limit on short')
    plt.legend()
    plt.xlabel('Risk in %')
    plt.ylabel('Return in %')
    plt.show()
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
