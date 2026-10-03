import marimo

__generated_with = "0.13.4"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # LASSO
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We wish to recover a sparse vector $x \in \mathbf{R}^n$ from measurements $y \in \mathbf{R}^m$. Our measurement model tells us that
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        $$
        y = Ax + v,
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        where $A \in \mathbf{R}^{m \times n}$ is a known matrix and $v \in \mathbf{R}^m$ is unknown measurement error.
        For our demonstration the entries of $v$ are sampled from the normal distribution with mean zero and
        standard deviation $\sigma$ (by default, $\sigma = 1$).

        We can first try to recover $x$ by solving the optimization problem
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        $$
        \begin{array}{ll} \text{minimize} & ||Ax - y||^2_2 + \gamma ||x||^2_2.\\
        \end{array}
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        This problem is called ridge regression.

        The code below defines $n$, $m$, $A$, $x$, and $y$. Use CVXPY to estimate $x$ from $y$ using ridge regression. Try multiple
        values of $\gamma$. Use the plotting code to compare the estimated $x$ with the true $x$.

        A more effective approach is to solve the LASSO problem
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        $$
        \begin{array}{ll} \text{minimize} & ||Ax - y||^2_2 + \gamma \|x\|_1.\\
        \end{array}
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        How many measurements $m$ are needed to find an accurate $x$ with ridge regression? How about with the LASSO?
        """
    )
    return


@app.cell
def _():
    # Ridge regression vs. LASSO to estimate sparse x.
    import numpy as np
    import scipy.linalg as la
    import scipy.sparse as sp
    import cvxpy as cp

    np.random.seed(1)

    n = 400
    m = 200
    true_x = 100 * sp.rand(n, 1, 0.1).toarray().flatten()
    A = np.random.randn(m, n)
    sigma = 1.0
    v = np.random.normal(0, sigma, m)
    y = A @ true_x + v


    x = cp.Variable(n)
    gamma = None  # set me! Initialize to 1.
    return cp, la, n, np, true_x, x


@app.cell
def _(cp, n, np, true_x, x):
    ridge_loss = None # set me
    ridge = cp.Problem(cp.Minimize(ridge_loss))
    ridge.solve(solver='CLARABEL')
    x_ridge = x.value

    lasso_loss = None # set me
    lasso = cp.Problem(cp.Minimize(lasso_loss))
    lasso.solve(solver='CLARABEL')
    x_lasso = x.value

    import matplotlib.pyplot as plt
    # '%matplotlib inline' command supported automatically in marimo
    plt.semilogy(range(n), np.sort(np.abs(true_x - x_ridge)),  label="ridge errors")
    plt.semilogy(range(n), np.sort(np.abs(true_x - x_lasso)),  label="lasso errors")
    plt.legend()
    plt.show()
    return (plt,)


@app.cell
def _(np):
    # Managing parallelism for cross-validation.
    import time
    import dask

    def get_x(gamma_val):
        # set the regularization parameter
        # gamma to gamma_val. Solve the problem
        # with the CLARABEL solver. Return the 
        # optimal "x".
        return None

    num_gamma = 30
    gamma_values = np.logspace(-4, 2, num_gamma)

    tic = time.time()
    xs_loop = [get_x(val) for val in gamma_values]
    toc = time.time()
    t_loop = toc - tic

    tic = time.time()
    dasklist = [dask.delayed(get_x)(val) for val in gamma_values]
    xs_dask = dask.compute(*dasklist, scheduler='processes')
    toc = time.time()
    t_dask = toc - tic
    return num_gamma, t_dask, t_loop, xs_dask, xs_loop


@app.cell
def _(la, num_gamma, plt, t_dask, t_loop, true_x, xs_dask, xs_loop):
    print(f'Time using a native loop \n\t{t_loop}')
    print(f'Time to solve using Dask parallelism \n\t{t_dask}')

    sol_diffs = [la.norm(xs_loop[i] - xs_dask[i]) for i in range(num_gamma)]
    print(f'Maximum discrepency between computed solutions \n\t{max(sol_diffs)}')

    sol_errors = [la.norm(xs_loop[i] - true_x) for i in range(num_gamma)]
    plt.plot(sol_errors)
    plt.show()
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
