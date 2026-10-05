import marimo

__generated_with = "0.13.4"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Hello, world!

        [Install CVXPY](https://www.cvxpy.org/install/index.html) and run the code on the [CVXPY landing page](https://www.cvxpy.org/).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Solve the following optimization problem using CVXPY:
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        $$
        \begin{array}{ll} \text{minimize} & |x| - 2\sqrt{y}\\
        \text{subject to} & 2 \geq e^x \\
        & x + y = 5,
        \end{array}
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        where $x,y \in \mathbf{R}$ are variables.

        Find the optimal values of $x$ and $y$.
        """
    )
    return


@app.cell
def _():
    # TODO: your code here
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
