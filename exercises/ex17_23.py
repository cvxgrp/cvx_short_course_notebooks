import marimo

__generated_with = "0.13.4"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Portfolio Rebalancing
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We consider the problem of rebalancing a portfolio of assets over multiple periods.
        We let $h_t \in \mathbb{R}^n$ denote the vector of our dollar value 
        holdings in $n$ assets,
        at the beginning of period $t$, for $t=1,\ldots, T$,
        with negative entries meaning short positions.
        We will work with the portfolio weight vector,
        defined as $w_t = h_t/(\mathbb{1}^Th_t)$, where
        we assume that $\mathbb{1}^T h_t>0$, *i.e.*, the total
        portfolio value is positive. 

        The *target portfolio weight vector*
        $w^\star$ is defined as the solution of the problem
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        $$
            \begin{array}{ll}
            \text{maximize}   & \mu^T w - \frac{\gamma}{2} w^T \Sigma w \\
            \text{subject to}
            & \mathbb{1}^Tw = 1,
            \end{array}
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        where $w\in \mathbb{R}^n$ is the variable, $\mu$ is the mean return,
        $\Sigma\in \mathbb{S}_{++}^n$ is the return covariance, and $\gamma>0$ is the
        risk aversion parameter.  The data $\mu$, $\Sigma$, and $\gamma$ are given.
        In words, the target weights maximize the risk-adjusted expected return.

        At the beginning of each period $t$ we are allowed to rebalance the
        portfolio by buying and selling assets.
        We call the post-trade portfolio weights $\tilde{w}_t$.
        They are found by solving the (rebalancing) problem
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        $$
            \begin{array}{ll}
            \text{maximize}   & \mu^T w - \frac{\gamma}{2} w^T \Sigma w - \kappa^T |w-w_t|\\
            \text{subject to}
            & \mathbb{1}^Tw = 1,
            \end{array}
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        with variable $w \in \mathbb{R}^n$,
        where $\kappa\in\mathbb{R}^n_+$ is the vector of (so-called linear) transaction costs
        for the assets. (For example, these could model bid/ask spread.)
        Thus, we choose the post-trade weights to maximize the risk-adjusted expected
        return, minus the transactions costs associated with rebalancing the portfolio.
        Note that the pre-trade weight vector $w_t$
        is known at the time we solve the problem.
        If we have $\tilde w_t = w_t$, it means that no rebalancing is done at
        the beginning of period $t$; we simply hold our current portfolio.
        (This happens if $w_t = w^\star$, for example.)

        After holding the rebalanced portfolio over the investment period,
        the dollar value of our portfolio becomes $h_{t+1} = \text{diag}(r_t) \tilde h_t$,
        where $r_t \in \mathbb{R}^n_{++}$ is the (random) vector of asset returns
        over period $t$, and $\tilde h_t$ is the post-trade portfolio
        given in dollar values (which you do not need to know).
        The next weight vector is then given by

        $$
            w_{t+1} =
            \frac{\text{diag}{(r_t)}\tilde{w}_t}{r_t^T\tilde{w}_t}.
        $$

        (If $r_t^T \tilde w_t \leq 0$, which means our portfolio has negative
        value after the investment period, we have gone bust, and all trading stops.)

        The standard model is that $r_t$ are IID random variables with
        mean and covariance $\mu$ and $\Sigma$,
        but this is not relevant in this problem.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Problem
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Starting from $w_1= w^\star$, compute a sequence of 
        portfolio weights $\tilde w_t$ for $t = 1,\ldots, T$.
        For each $t$, find $\tilde w_t$ 
        by solving the rebalancing problem (with $w_{t}$ a known constant);
        then generate a vector of returns $r_t$ (using our supplied function)
        to compute $w_{t+1}$
        (The sequence of weights is random, so the results won't be the 
        same each time you run your script.   But they should look similar.)

        Report the fraction of periods in which the solution has only zero (or negligible)
        trades, defined as 

        $$
        \|\tilde w_t - w_t \|_\infty \leq 10^{-3}.
        $$

        Plot the sequence $\tilde w_t$ for $t = 1, 2,\ldots, T$.

        We provide below the data,
        a function to generate a (random) vector $r_t$ of market returns, and
        the code to plot the sequence $\tilde w_t$. (The plotting code also
        draws a dot for every non-negligible trade.) 

        Carry this out for two values of $\kappa$,
        $\kappa = \kappa_1$ and $\kappa = \kappa_2$. 
        Briefly comment on what you observe.
        """
    )
    return


@app.cell
def _():
    # data for multiperiod portfolio rebalancing problem
    import numpy as np

    T = 100
    n = 5
    gamma = 8.0
    threshold = 0.001
    Sigma = np.array(
        [
            [1.512e-02, 1.249e-03, 2.762e-04, -5.333e-03, -7.938e-04],
            [1.249e-03, 1.030e-02, 6.740e-05, -1.301e-03, -1.937e-04],
            [2.762e-04, 6.740e-05, 1.001e-02, -2.877e-04, -4.283e-05],
            [-5.333e-03, -1.301e-03, -2.877e-04, 1.556e-02, 8.271e-04],
            [-7.938e-04, -1.937e-04, -4.283e-05, 8.271e-04, 1.012e-02],
        ]
    )
    mu = np.array([1.02, 1.028, 1.01, 1.034, 1.017])
    kappa_1 = np.full(n, 0.002)
    kappa_2 = np.full(n, 0.004)

    ## Generate returns
    # call this function to generate a vector r of market returns
    def generate_returns():
        return np.random.multivariate_normal(mu, Sigma)

    return T, n, np, threshold


@app.cell
def _():
    import cvxpy as cp

    # TODO: your code here
    # You must provide three objects:
    # - ws: np.array of size T x n,
    #       the post-trade weights w_t_tilde;
    # - us: np.array of size T x n,
    #       the trades at each period: w_t_tilde - w_t;
    # - w_star: np.array of size n,
    #       the "target" solution w_star.
    return


@app.cell
def _(T, n, np, threshold, us, w_star, ws):
    ## Plotting code
    import matplotlib.pyplot as plt

    colors = ["b", "r", "g", "c", "m"]
    plt.figure(figsize=(13, 5))
    for j in range(n):
        plt.plot(range(T), ws[:, j], colors[j])
        plt.axhline(w_star[j], color=colors[j], linestyle="--")
        non_zero_trades = np.abs(us[:, j]) > threshold
        plt.plot(np.arange(T)[non_zero_trades], ws[non_zero_trades, j], colors[j] + "o")
    plt.ylabel("post-trade weights")
    plt.xlabel("period $t$")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
