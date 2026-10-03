import marimo

__generated_with = "0.13.4"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # DCP analysis in CVXPY
        """
    )
    return


@app.cell
def _():
    import cvxpy as cp


    x = cp.Variable()
    y = cp.Variable()
    expr = cp.quad_over_lin(x - y, 1 - cp.maximum(x, y))
    print("expression curvature =", expr.curvature)
    print("expression sign =", expr.sign)
    print("expression is DCP?", expr.is_dcp())
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
