import marimo

__generated_with = "0.13.4"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Portfolio optimization""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Portfolio allocation vector

    In this example we show how to do portfolio optimization using CVXPY.
    We begin with the basic definitions.
    In portfolio optimization we have some amount of money to invest in any of $n$ different assets. 
    We choose what fraction $w_i$ of our money to invest in each asset $i$, $i=1, \ldots, n$.

    We call $w\in {\bf R}^n$ the *portfolio allocation vector*.
    We of course have the constraint that ${\mathbf 1}^T w =1$.
    The allocation $w_i<0$ means a *short position* in asset $i$, or that we borrow shares to sell now that we must replace later.
    The allocation $w \geq 0$ is a *long only* portfolio.
    The quantity
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    $$
    \|w \|_1 = {\mathbf 1}^T w_+ + {\mathbf 1}^T w_-
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""is known as *leverage*.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Asset returns

    We will only model investments held for one period. The initial prices are $p_i > 0$. The end of period prices are $p_i^+ >0$. The asset (fractional) returns are $r_i = (p_i^+-p_i)/p_i$. The porfolio (fractional) return is $R = r^Tw$.

    A common model is that $r$ is a random variable with mean ${\bf E}r = \mu$ and covariance ${\bf E{(r-\mu)(r-\mu)^T}} = \Sigma$.
    It follows that $R$ is a random variable with ${\bf E}R = \mu^T w$ and ${\bf var}(R) = w^T\Sigma w$.
    ${\bf E}R$ is the (mean) *return* of the portfolio. ${\bf var}(R)$ is the *risk* of the portfolio.
    (Risk is also sometimes given as ${\bf std}(R) = \sqrt{{\bf var}(R)}$.)

    Portfolio optimization has two competing objectives: high return and low risk.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Classical (Markowitz) portfolio optimization

    Classical (Markowitz) portfolio optimization solves the optimization problem
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    $$
    \begin{array}{ll} \text{maximize} & \mu^T w - \gamma w^T\Sigma w\\
    \text{subject to} & {\bf 1}^T w = 1, \quad w \in {\cal W},
    \end{array}
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    where $w \in {\bf R}^n$ is the optimization variable, $\cal W$ is a set of allowed portfolios (e.g., ${\cal W} = {\bf R}_+^n$ for a long only portfolio), and $\gamma >0$ is the *risk aversion parameter*.

    The objective $\mu^Tw - \gamma w^T\Sigma w$ is the *risk-adjusted return*. Varying $\gamma$ gives the optimal *risk-return trade-off*.
    We can get the same risk-return trade-off by fixing return and minimizing risk.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Example

    In the following code we compute and plot the optimal risk-return trade-off for $10$ assets, restricting ourselves to a long only portfolio.
    """
    )
    return


@app.cell
def _():
    # Generate data for long only portfolio optimization.
    import numpy as np

    np.random.seed(1)
    n = 10
    mu = np.abs(np.random.randn(n))
    Sigma = np.random.randn(n, n)
    Sigma = Sigma.T @ Sigma
    return Sigma, mu, n, np


@app.cell
def _(Sigma, mu, n):
    # Long only portfolio optimization.
    import cvxpy as cp

    w = cp.Variable(n)
    gamma = cp.Parameter(nonneg=True)
    ret = mu @ w
    risk = cp.quad_form(w, Sigma)
    prob = cp.Problem(cp.Maximize(ret - gamma * risk), [cp.sum(w) == 1, w >= 0])
    return cp, gamma, prob, ret, risk, w


@app.cell
def _(gamma, np, prob, ret, risk):
    _SAMPLES = 100
    risk_data = np.zeros(_SAMPLES)
    ret_data = np.zeros(_SAMPLES)
    gamma_vals = np.logspace(-2, 3, num=_SAMPLES)
    for _i, _gamma_val in enumerate(gamma_vals):
        gamma.value = _gamma_val
        prob.solve()
        risk_data[_i] = np.sqrt(risk.value)
        ret_data[_i] = ret.value
    return gamma_vals, ret_data, risk_data


@app.cell
def _(Sigma, gamma_vals, mu, np, ret_data, risk_data):
    import matplotlib.pyplot as plt

    markers_on = [29, 40]
    fig, ax = plt.subplots()
    plt.plot(risk_data, ret_data, 'g-')
    for marker in markers_on:
        plt.plot(risk_data[marker], ret_data[marker], 'bs')
        ax.annotate(rf'$\gamma = {gamma_vals[marker]:.2f}$', xy=(risk_data[marker] + 0.08, ret_data[marker] - 0.03))
    plt.plot(np.sqrt(np.diag(Sigma)), mu, 'ro')
    plt.xlabel('Standard deviation')
    plt.ylabel('Return')
    plt.show()
    return markers_on, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We plot below the return distributions for the two risk aversion values marked on the trade-off curve.
    Notice that the probability of a loss is near 0 for the low risk value and far above 0 for the high risk value.
    """
    )
    return


@app.cell
def _(gamma, gamma_vals, markers_on, np, plt, prob, ret, risk):
    import scipy.stats as spstats

    x = np.linspace(-2, 5, 1000)
    plt.figure()
    for _idx in markers_on:
        gamma.value = gamma_vals[_idx]
        prob.solve()
        plt.plot(x, spstats.norm.pdf(x, ret.value, np.sqrt(risk.value)), label=rf'$\gamma = {gamma.value:.2f}$')
    plt.xlabel('Return')
    plt.ylabel('Density')
    plt.legend(loc='upper right')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Portfolio constraints

    There are many other possible portfolio constraints besides the long only constraint. With no constraint (${\cal W} = {\bf R}^n$), the optimization problem has a simple analytical solution. We will look in detail at a *leverage limit*, or the constraint that $\|w \|_1 \leq L^\mathrm{max}$.


    Another interesting constraint is the *market neutral* constraint $m^T \Sigma w =0$, where $m_i$ is the capitalization of asset $i$.
    $M = m^Tr$ is the *market return*, and $m^T \Sigma w = {\bf cov}(M,R)$.
    The market neutral constraint ensures that the portfolio return is uncorrelated with the market return.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Example

    In the following code we compute and plot optimal risk-return trade-off curves for leverage limits of 1, 2, and 4.
    Notice that more leverage increases returns and allows greater risk.
    """
    )
    return


@app.cell
def _(cp, gamma, ret, risk, w):
    Lmax = cp.Parameter(nonneg=True)
    prob_leverage = cp.Problem(cp.Maximize(ret - gamma * risk), [cp.sum(w) == 1, cp.norm(w, 1) <= Lmax])
    return Lmax, prob_leverage


@app.cell
def _(Lmax, gamma, gamma_vals, np, prob_leverage, ret, risk):
    L_vals = [1, 2, 4]
    risk_data_leverage = np.zeros((len(L_vals), len(gamma_vals)))
    ret_data_leverage = np.zeros((len(L_vals), len(gamma_vals)))
    for _k, _L_val in enumerate(L_vals):
        Lmax.value = _L_val
        for _i, _gamma_val in enumerate(gamma_vals):
            gamma.value = _gamma_val
            prob_leverage.solve()
            risk_data_leverage[_k, _i] = np.sqrt(risk.value)
            ret_data_leverage[_k, _i] = ret.value
    return L_vals, ret_data_leverage, risk_data_leverage


@app.cell
def _(
    L_vals,
    Sigma,
    mu,
    np,
    plt,
    ret_data_leverage,
    risk_data_leverage,
    w_vals,
):
    for _idx, _L_val in enumerate(L_vals):
        plt.plot(risk_data_leverage[_idx], ret_data_leverage[_idx], label=rf'$L^{{\max}}$ = {_L_val}')
    for _w_val in w_vals:
        plt.plot(np.sqrt(_w_val @ Sigma @ _w_val), mu @ _w_val, 'bs')
    plt.xlabel('Standard deviation')
    plt.ylabel('Return')
    plt.legend(loc='lower right')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We next examine the points on each trade-off curve where $w^T\Sigma w = 2$.
    We plot the amount of each asset held in each portfolio as bar graphs. (Negative holdings indicate a short position.)
    Notice that some assets are held in a long position for the low leverage portfolio but in a short position in the higher leverage portfolios.
    """
    )
    return


@app.cell
def _(Lmax, cp, ret, risk, w):
    prob_risk_limit = cp.Problem(cp.Maximize(ret), [cp.sum(w) == 1, cp.norm(w, 1) <= Lmax, risk <= 2])
    return (prob_risk_limit,)


@app.cell
def _(L_vals, Lmax, prob_risk_limit, w):
    w_vals = []
    for _L_val in L_vals:
        Lmax.value = _L_val
        prob_risk_limit.solve()
        w_vals.append(w.value)
    return (w_vals,)


@app.cell
def _(L_vals, mu, n, np, plt, w_vals):
    colors = ['b', 'g', 'r']
    indices = np.argsort(mu)
    for _idx, _L_val in enumerate(L_vals):
        plt.bar(np.arange(1, n + 1) + 0.25 * _idx - 0.375, w_vals[_idx][indices], color=colors[_idx], label=rf'$L^{{\max}}$ = {_L_val}', width=0.25)
    plt.ylabel('$w_i$', fontsize=16)
    plt.xlabel('$i$', fontsize=16)
    plt.xlim([1 - 0.375, n + 0.375])
    plt.xticks(np.arange(1, n + 1))
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Variations

    There are many more variations of classical portfolio optimization. We might require that $\mu^T w \geq R^\mathrm{min}$ and minimize $w^T \Sigma w$ or $\|\Sigma ^{1/2} w\|_2$.
    We could include the (broker) cost of short positions as the penalty $s^T (w)_-$ for some $s \geq 0$.
    We could include transaction costs (from a previous portfolio $w^\mathrm{prev}$) as the penalty
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    $$
    \kappa ^T |w-w^\mathrm{prev}|^\eta, \quad
    \kappa \geq 0.
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Common values of $\eta$ are $\eta =1, ~ 3/2, ~2$.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Factor covariance model

    A particularly common and useful variation is to model the covariance matrix $\Sigma$ as a factor model
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    $$
    \Sigma  =  F \tilde \Sigma F^T + D,
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    where  $F \in {\bf R}^{n \times k}$, $k \ll n$ is the *factor loading matrix*. $k$ is the number of factors (or sectors) (typically 10s). $F_{ij}$ is the loading of asset $i$ to factor $j$.
    $D$ is a diagonal matrix; $D_{ii}>0$ is the *idiosyncratic risk*. $\tilde \Sigma > 0$ is the *factor covariance matrix*.

    $F^Tw \in {\bf R}^k$ gives the portfolio *factor exposures*. A portfolio is *factor $j$ neutral* if $(F^Tw)_j=0$.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Portfolio optimization with factor covariance model

    Using the factor covariance model, we frame the portfolio optimization problem as
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    $$
    \begin{array}{ll} \text{maximize} & \mu^T w - \gamma \left(f^T \tilde \Sigma f  + w^TDw \right) \\
    \text{subject to} & {\bf 1}^T w = 1, \quad f=F^Tw\\
    & w \in {\cal W}, \quad f \in {\cal F},
    \end{array}
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    where the variables are the allocations $w \in {\bf R}^n$ and factor exposures $f\in {\bf R}^k$ and $\cal F$ gives the factor exposure constraints.

    Using the factor covariance model in the optimization problem has a computational advantage. The solve time is $O(nk^2)$ versus $O(n^3)$ for the standard problem.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Example

    In the following code we generate and solve a portfolio optimization problem with 30 factors and 3000 assets.
    We set the leverage limit $=2$ and $\gamma=0.1$.

    We solve the problem both with the covariance given as a single matrix and as a factor model.
    """
    )
    return


@app.cell
def _(np):
    n_large = 600
    m = 30
    np.random.seed(1)
    mu_large = np.abs(np.random.randn(n_large))
    Sigma_tilde = np.random.randn(m, m)
    Sigma_tilde = Sigma_tilde.T @ Sigma_tilde
    d = np.random.uniform(0, 0.9, size=n_large)
    F = np.random.randn(n_large, m)
    return F, Sigma_tilde, d, m, mu_large, n_large


@app.cell
def _(F, Sigma_tilde, cp, d, m, mu_large, n_large, np):
    w_large = cp.Variable(n_large)
    f = cp.Variable(m)
    gamma_large = cp.Parameter(nonneg=True)
    Lmax_large = cp.Parameter(nonneg=True)
    ret_large = mu_large @ w_large
    risk_factor = cp.quad_form(f, Sigma_tilde) + cp.sum_squares(cp.multiply(np.sqrt(d), w_large))
    prob_factor = cp.Problem(cp.Maximize(ret_large - gamma_large * risk_factor), [cp.sum(w_large) == 1, f == F.T @ w_large, cp.norm(w_large, 1) <= Lmax_large])
    Lmax_large.value = 2
    gamma_large.value = 0.1
    prob_factor.solve(verbose=True)
    return Lmax_large, gamma_large, prob_factor, ret_large, w_large


@app.cell
def _(F, Lmax_large, Sigma_tilde, cp, d, gamma_large, np, ret_large, w_large):
    risk_full = cp.quad_form(w_large, cp.psd_wrap(F @ Sigma_tilde @ F.T + np.diag(d)))
    prob_full = cp.Problem(cp.Maximize(ret_large - gamma_large * risk_full), [cp.sum(w_large) == 1, cp.norm(w_large, 1) <= Lmax_large])
    prob_full.solve(verbose=True, max_iter=100000)
    return (prob_full,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We expect the factor model to be at least 20 times faster than the standard model as
    $n^2 = 20 n m$ where $n$ is the number of assets and $m$ is the number of factors.
    This estimate does not yet reflect a smaller number of iterations required for the factor model.
    """
    )
    return


@app.cell
def _(prob_factor, prob_full):
    print(f'Factor model solve time = {prob_factor.solver_stats.solve_time}')
    print(f'Single model solve time = {prob_full.solver_stats.solve_time}')
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
