import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # DCP analysis
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In this exercise, you will fix optimization problems that break the DCP rules by identifying the DCP error and then rewriting the problem.
    """)
    return


@app.cell
def _():
    import cvxpy as cp
    import numpy as np

    return cp, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Problem 1.

    $$
    \begin{array}{ll}
    \text{minimize} & \sqrt{x^2 + 1},
    \end{array}
    $$

    with variable $x \in \mathbf{R}$.
    """)
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
    mo.md(r"""
    ## Problem 2.

    $$
    \begin{array}{ll}
    \text{minimize} & x + 2 \\
    \text{subject to} & 5 = 2 / x \\
    & x > 0,
    \end{array}
    $$

    with variable $x \in \mathbf{R}$.
    """)
    return


@app.cell
def _(cp):
    _x = cp.Variable()
    _prob = cp.Problem(cp.Minimize(_x + 2), [5 == 2 * cp.inv_pos(_x)])
    _prob.solve()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Problem 3.

    $$
    \begin{array}{ll}
    \text{minimize} & x + 2 \\
    \text{subject to} & 5 \leq 2 / x^2,
    \end{array}
    $$

    with variable $x \in \mathbf{R}$.
    """)
    return


@app.cell
def _(cp):
    _x = cp.Variable()
    _prob = cp.Problem(cp.Minimize(_x + 2), [5 <= 2 * _x ** (-2)])
    _prob.solve()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Problem 4.

    $$
    \begin{array}{ll}
    \text{minimize} & 1 / x \\
    \text{subject to} & 0 \leq x^2 / y \\
    & y \geq 1 \\
    & x > 0,
    \end{array}
    $$

    with variables $x, y \in \mathbf{R}$.
    """)
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
    mo.md(r"""
    ## Problem 5.

    $$
    \begin{array}{ll}
    \text{minimize} & x + 2 \\
    \text{subject to} & \exp(2x) + \exp(3x) \leq \exp(5x),
    \end{array}
    $$

    with variable $x \in \mathbf{R}$.
    """)
    return


@app.cell
def _(cp):
    _x = cp.Variable()
    _prob = cp.Problem(cp.Minimize(_x + 2), [cp.exp(2 * _x) + cp.exp(3 * _x) <= cp.exp(5 * _x)])
    _prob.solve()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bonus Problem 1.

    $$
    \begin{array}{ll}
    \text{minimize} & (\max\{x, 4\} - 3)^2 \\
    \text{subject to} & x \geq 1,
    \end{array}
    $$

    with variable $x \in \mathbf{R}$.
    """)
    return


@app.cell
def _(cp):
    _x = cp.Variable()
    _prob = cp.Problem(cp.Minimize((cp.maximum(_x, 4) - 3) ** 2), [_x >= 1])
    _prob.solve()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bonus Problem 2.

    $$
    \begin{array}{ll}
    \text{minimize} & \sum_{i=1}^m c_i \dfrac{x_i}{u_i - x_i} \\
    \text{subject to} & x \prec u,
    \end{array}
    $$

    with variable $x \in \mathbf{R}^m$, where $c, u \in \mathbf{R}^m_+$ are given.
    """)
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
