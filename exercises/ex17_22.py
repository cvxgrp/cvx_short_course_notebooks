import marimo

__generated_with = "0.13.4"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Risk budget allocation.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Suppose an amount $x_i>0$ is invested in $n$ assets, labeled $i=1,..., n$,
    with asset return covariance matrix $\Sigma \in \mathcal{S}_{++}^n$.
    We define the *risk* of the investments as the standard deviation
    of the total return, $R(x) = (x^T\Sigma x)^{1/2}$.

    We define the (relative) *risk contribution* of asset $i$
    (in the portfolio $x$) as
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    $$
    \rho_i = \frac{\partial \log R(x)}{\partial \log x_i} =
    \frac{\partial R(x)}{R(x)} \frac{x_i}{\partial x_i}, \quad i=1, \ldots, n.
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Thus $\rho_i$ gives the fractional increase in risk per fractional increase
    in investment $i$.
    We can express the risk contributions as
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    $$
    \rho_i = \frac{x_i (\Sigma x)_i} {x^T\Sigma x}, \quad i=1, \ldots, n,
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    from which we see that $\sum_{i=1}^n \rho_i = 1$.
    For general $x$, we can have $\rho_i <0$, which means that a small increase
    in investment $i$ decreases the risk.
    Desirable investment choices have $\rho_i>0$, in which case we can
    interpret $\rho_i$ as the fraction of the total risk contributed by
    the investment in asset $i$.
    Note that the risk contributions are homogeneous of degree zero, i.e.,
    scaling $x$ by a positive constant does not affect $\rho_i$.

    In the *risk budget allocation problem*, we are given $\Sigma$ and a
    set of desired risk contributions $\rho_i^\mathrm{des}>0$ with
    $\bf{1}^T \rho^\mathrm{des}=1$; the goal is to find an investment mix
    $x\succ 0$, $\bf{1}^Tx =1$, with these risk contributions.
    When $\rho^\mathrm{des} = (1/n)\bf{1}$, the problem is to find an
    investment mix that achieves so-called *risk parity*.

    ## (a)

    Explain how to solve the risk budget allocation problem using convex
    optimization.

    *Hint.*
    Minimize $(1/2)x^T\Sigma x - \sum_{i=1}^n \rho_i^\mathrm{des} \log x_i$.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## (b)

    Find the investment mix that achieves risk parity for the
    return covariance matrix $\Sigma$ below.
    """
    )
    return


@app.cell
def _():
    import numpy as np
    import cvxpy as cp

    Sigma = np.array(
        [
            [6.1, 2.9, -0.8, 0.1],
            [2.9, 4.3, -0.3, 0.9],
            [-0.8, -0.3, 1.2, -0.7],
            [0.1, 0.9, -0.7, 2.3],
        ]
    )
    rho = np.ones(4) / 4

    # TODO: your code here
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
