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
    import cvxpy as cp

    np.random.seed(1)

    n = 400
    m = 200
    true_x = np.zeros(n)
    true_x[np.random.choice(n, n // 10, replace=False)] = 100 * np.random.rand(n // 10)
    A = np.random.randn(m, n)
    sigma = 1.0
    v = np.random.normal(0, sigma, m)
    y = A @ true_x + v

    x = cp.Variable(n)
    # TODO: your code here
    # Define gamma as a nonnegative CVXPY parameter with initial value 1.
    gamma = None
    return cp, n, np, true_x, x


@app.cell
def _(cp, n, np, true_x, x):
    ridge_loss = None  # TODO: your code here
    ridge = cp.Problem(cp.Minimize(ridge_loss))
    ridge.solve(solver='CLARABEL')
    x_ridge = x.value

    lasso_loss = None  # TODO: your code here
    lasso = cp.Problem(cp.Minimize(lasso_loss))
    lasso.solve(solver='CLARABEL')
    x_lasso = x.value

    import matplotlib.pyplot as plt

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
        # TODO: your code here
        # Set the regularization parameter gamma to gamma_val, solve the
        # problem with the CLARABEL solver, and return the optimal x.
        return None

    num_gamma = 30
    gamma_values = np.logspace(-4, 2, num_gamma)

    tic = time.perf_counter()
    xs_loop = [get_x(val) for val in gamma_values]
    toc = time.perf_counter()
    t_loop = toc - tic

    tic = time.perf_counter()
    dasklist = [dask.delayed(get_x)(val) for val in gamma_values]
    xs_dask = dask.compute(*dasklist, scheduler='processes')
    toc = time.perf_counter()
    t_dask = toc - tic
    return t_dask, t_loop, xs_dask, xs_loop


@app.cell
def _(np, plt, t_dask, t_loop, true_x, xs_dask, xs_loop):
    print(f'Time using a native loop \n\t{t_loop}')
    print(f'Time to solve using Dask parallelism \n\t{t_dask}')

    sol_diffs = np.linalg.norm(np.array(xs_loop) - np.array(xs_dask), axis=1)
    print(f'Maximum discrepancy between computed solutions \n\t{sol_diffs.max()}')

    sol_errors = np.linalg.norm(np.array(xs_loop) - true_x, axis=1)
    plt.plot(sol_errors)
    plt.show()
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
