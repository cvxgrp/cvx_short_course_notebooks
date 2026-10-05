import marimo

__generated_with = "0.13.4"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Trade off curves
        """
    )
    return


@app.cell
def _():
    # Generate data for problem.
    import numpy as np

    np.random.seed(1)
    m = 25
    n = 10
    A = np.random.randn(m, n)
    b = np.random.randn(m)
    return A, b, n, np


@app.cell
def _(A, b, n):
    # Form and solve problem.
    import cvxpy as cp


    gamma = cp.Parameter(nonneg=True)
    gamma.value = 1
    x = cp.Variable(n)
    cost = cp.sum_squares(A @ x - b) + gamma * cp.norm(x, 1)
    prob = cp.Problem(cp.Minimize(cost), [cp.norm(x, "inf") <= 1])
    opt_val = prob.solve()
    solution = x.value
    return gamma, prob, x


@app.cell
def _(gamma, np, prob, x):
    # For loop style trade-off curve.
    gamma_vals = np.logspace(-4, 2, 100)
    x_values = []
    for val in gamma_vals:
        gamma.value = val
        prob.solve()
        x_values.append(x.value)
    return gamma_vals, x_values


@app.cell
def _(gamma, gamma_vals, prob, x):
    # Parallel style trade-off curve.

    # Use tools for parallelism in standard library.
    import dask

    # Function maps gamma value to optimal x.
    def get_x(gamma_value):
        gamma.value = gamma_value
        prob.solve()
        return x.value

    dasklist = [dask.delayed(get_x)(val) for val in gamma_vals]
    xs_dask = dask.compute(*dasklist)
    return


@app.cell
def _(gamma_vals, n, x_values):
    # Plot regularization path.
    import matplotlib.pyplot as plt

    for i in range(n):
        plt.plot(gamma_vals, [xi[i] for xi in x_values])
    plt.xlabel(r"$\gamma$", fontsize=16)
    plt.ylabel(r"$x_i$", fontsize=16)
    plt.xscale("log")
    plt.title(r"Entries of $x$ versus $\gamma$", fontsize=16)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
