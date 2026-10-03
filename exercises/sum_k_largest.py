import marimo

__generated_with = "0.13.4"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # A toy mixed-norm regression problem
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let $f_k(x)$ be the sum of the $k$ largest entries of $|x|$.

        It turns out that this function is a norm! Special cases:
         * $f_1(x) = \|x\|_{\infty}$
         * $f_n(x) = \|x\|_1$.

        Consider the minimum-norm regression problem:

        $\qquad \operatorname{minimize} ~f_k(x) + \gamma f_n(x) ~\text{ s.t. }~ A x = y$.

        For a wide $m \times n$ matrix $A$, an $m$-vector $y$, and a tuning parameter $\gamma$.

        The larger values of $x$ are "more penalized" as $\gamma$ gets smaller.
        """
    )
    return


@app.cell
def _():
    import cvxpy as cp
    import itertools

    def sum_abs_largest(expr, k):
        # implement me!
        return None
    return cp, sum_abs_largest


@app.cell
def _(cp):
    import numpy as np
    import scipy.linalg as la
    import scipy.sparse as sp
    import time

    def make_problem_data(n, m, true_k):
        np.random.seed(1)
        x_ref = np.zeros(n)
        idxs = np.random.choice(np.arange(n), true_k, replace=False)
        x_ref[idxs] = 10 + 100*np.random.rand(true_k)
        A = np.random.randn(m, n)
        v = np.random.normal(0, 1.0, m)
        v /= (m*la.norm(v))
        y = A @ x_ref + v
        return A, x_ref, y

    def make_and_solve(loss, x, A, y):
        prob = cp.Problem(cp.Minimize(loss), [A @ x == y])
        tic = time.time()
        prob.solve(solver='CLARABEL')
        toc = time.time()
        t = toc - tic
        x_est = x.value
        return prob, x_est, t
    return la, make_and_solve, make_problem_data


@app.cell
def _(cp, make_problem_data):
    n, m, true_k = 14, 7, 4
    A, x_ref, y = make_problem_data(n, m, true_k)
    x = cp.Variable(n)
    gamma = cp.Parameter(pos=True)
    gamma.value = 1e-3
    k = 3
    return A, gamma, k, x, y


@app.cell
def _(A, cp, gamma, k, la, make_and_solve, sum_abs_largest, x, y):
    loss1 = sum_abs_largest(x, k) + gamma * cp.norm(x, 1)
    prob1, x1, t1 = make_and_solve(loss1, x, A, y)
    print(f'Solve time (home brewed)\n\t{t1}')

    loss2 = cp.sum_largest(cp.abs(x), k) + gamma * cp.norm(x, 1)
    prob2, x2, t2 = make_and_solve(loss2, x, A, y)
    print(f'Solve time (cvxpy atom)\n\t{t2}')

    disc = la.norm(x1 - x2) / min(la.norm(x1), la.norm(x2))
    print(f'Discrepency between two solutions\n\t{disc}')
    return prob1, prob2


@app.cell
def _(prob1, prob2):
    import matplotlib.pyplot as plt
    # '%matplotlib inline' command supported automatically in marimo

    A1 = prob1.get_problem_data(solver='SCS')[0]['A']
    plt.spy(A1, aspect='auto')
    plt.show()

    A2 = prob2.get_problem_data(solver='SCS')[0]['A']
    plt.spy(A2, aspect='auto')
    plt.show()
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
