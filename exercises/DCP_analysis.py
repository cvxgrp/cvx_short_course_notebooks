import marimo

__generated_with = "0.13.4"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# DCP analysis""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""In this exercise, you will fix optimization problems that break the DCP rules by identifying the DCP error and then rewriting the problem.""")
    return


@app.cell
def _():
    import cvxpy as cp
    import numpy as np
    return cp, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Problem 1.

    $\min\{  \sqrt{x^2 + 1 } : x \in \mathbf{R} \}$
    """
    )
    return


@app.cell
def _(cp):
    _x = cp.Variable()
    _cost = (_x ** 2 + 1) ** 0.5
    _prob = cp.Problem(cp.Minimize(_cost))
    _prob.solve()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Problem 2.

    $\min\{x + 2 \,:\, 5 = 2 / x,~~  x > 0 \}$
    """
    )
    return


@app.cell
def _(cp):
    _x = cp.Variable()
    _prob = cp.Problem(cp.Minimize(_x + 2), [5 == 2 * cp.inv_pos(_x)])
    _prob.solve()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Problem 3.
    $\min\{ x + 2 \,:\, 5 \leq 2 / x^2,~~ x \in \mathbf{R} \}$
    """
    )
    return


@app.cell
def _(cp):
    _x = cp.Variable()
    _prob = cp.Problem(cp.Minimize(_x + 2), [5 <= 2 * _x ** (-2)])
    _prob.solve()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Problem 4.
    $\min\{ 1/ x \,:\, 0 \leq x^2 / y, ~~ y \geq 1, ~~ x > 0 \}$
    """
    )
    return


@app.cell
def _(cp):
    _x = cp.Variable()
    y = cp.Variable()
    _prob = cp.Problem(cp.Minimize(cp.pos(_x)), [0 <= cp.quad_over_lin(_x, y), y >= 1])
    _prob.solve()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Problem 5.
    $\min\{ x + 2 \,:\, \exp(2x) + \exp(3x) \leq \exp(5x)  \}$
    """
    )
    return


@app.cell
def _(cp):
    _x = cp.Variable()
    _prob = cp.Problem(cp.Minimize(_x + 2), [cp.exp(2 * _x) + cp.exp(3 * _x) <= cp.exp(5 * _x)])
    _prob.solve()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Bonus Problem 1.
    $\min\{ -(\max\{x, 4\} - 3)^2 \,:\, x \geq 1 \}$
    """
    )
    return


@app.cell
def _(cp):
    _x = cp.Variable()
    _prob = cp.Problem(cp.Maximize(-(cp.maximum(_x, 4) - 3) ** 2), [_x >= 1])
    _prob.solve()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Bonus Problem 2.

    $\min\left\{ \sum_{i=1}^m c_i \frac{x_i}{u_i - x_i} \,:\, ~  u > x,~~ x \in \mathbf{R}^m \right\}$

    where $c$ and $u$ are nonnegative vectors.
    """
    )
    return


@app.cell
def _(cp, np):
    m = 10
    np.random.seed(1)
    c = np.random.randn(m)
    c = np.abs(c)
    u = np.random.randn(m)
    u = np.abs(u)
    _x = cp.Variable(m)
    _cost = sum([c[i] * _x[i] * cp.inv_pos(u[i] - _x[i]) for i in range(m)])
    _prob = cp.Problem(cp.Minimize(_cost))
    _prob.solve()
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
