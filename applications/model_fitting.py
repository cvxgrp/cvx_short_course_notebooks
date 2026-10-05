import marimo

__generated_with = "0.25.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Logistic regression
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Binary classification

    In this example we do logistic regression in CVXPY.
    We are given data $(x_i, y_i)$, $i=1,\ldots,m$, where $x_i \in {\bf R}^n$ is a *feature vector*
    and $y_i \in \{0,1\}$ is a (boolean) *label*.
    Our goal is to construct a *classifier* that predicts the label $y$ of a new feature vector $x$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Probabilistic model

    We model each data point $(x_i, y_i)$ as an outcome of a random variable $(X, Y)$,
    where $X$ takes values in ${\bf R}^n$ and $Y$ takes values in $\{0,1\}$.
    We model the conditional distribution of the label given the features as

    $$
    \mathop{\bf Prob}(Y = 1 \mid X = x) = \sigma(\theta^T x), \qquad \sigma(u) = \frac{1}{1 + e^{-u}},
    $$

    where $\theta \in {\bf R}^n$ is the model parameter and $\sigma$ is the *logistic* (sigmoid) function.
    Since the label is binary, $\mathop{\bf Prob}(Y = 0 \mid X = x) = 1 - \sigma(\theta^T x) = \sigma(-\theta^T x)$.

    Given $\theta$, we predict $\hat y = 1$ if $\theta^T x \geq 0$, i.e., if the probability of
    $Y = 1$ is at least $1/2$, and $\hat y = 0$ otherwise.
    The decision boundary is the hyperplane $\theta^T x = 0$.
    An intercept term is obtained by appending a constant feature $1$ to $x$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Maximum likelihood estimation

    Assuming the data points are independent, the likelihood of $\theta$ is

    $$
    \prod_{i=1}^m \sigma(\theta^T x_i)^{y_i} \left(1 - \sigma(\theta^T x_i)\right)^{1 - y_i}.
    $$

    Using $\log \sigma(u) = u - \log(1 + e^u)$ and $\log(1 - \sigma(u)) = -\log(1 + e^u)$,
    the negative log-likelihood simplifies to

    $$
    \ell(\theta) = \sum_{i=1}^m \left( \log\left(1 + \exp(\theta^T x_i)\right) - y_i \theta^T x_i \right).
    $$

    Maximum likelihood estimation therefore boils down to the optimization problem

    $$
    \begin{array}{ll}
    \text{minimize} & \sum_{i=1}^m \log\left(1 + \exp(\theta^T x_i)\right) - y_i \theta^T x_i,
    \end{array}
    $$

    with variable $\theta$. Each term is a log-sum-exp of $(0, \theta^T x_i)$ minus an affine function of $\theta$,
    so the problem is convex.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Regularization

    A standard variant is to add a regularizer and solve

    $$
    \begin{array}{ll}
    \text{minimize} & (1/m) \ell(\theta) + \lambda \|\theta\|_1,
    \end{array}
    $$

    where $\lambda \geq 0$ is a parameter.
    The $\ell_1$ term encourages a sparse $\theta$, i.e., it selects features.
    Increasing $\lambda$ trades off fit to the training data for sparsity.
    The problem is still convex.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Example

    In the following code we generate data from the model itself.
    We draw features $x_i \sim \mathcal N(0, I)$ in ${\bf R}^{200}$ and append a constant feature $1$ for the intercept.
    We choose a sparse $\theta^\mathrm{true}$ with $5$ nonzero entries and draw the labels as
    $y_i \sim \mathrm{Bernoulli}(\sigma((\theta^\mathrm{true})^T x_i))$.
    We fit $\ell_1$-regularized logistic regression, without regularizing the intercept.
    We train on $m=200$ examples and test on $1000$ examples, and plot the train and test error as we vary $\lambda$.
    """)
    return


@app.cell
def _():
    import numpy as np

    np.random.seed(1)
    n = 200
    m = 200
    TEST = 1000
    NONZEROS = 5
    theta_true = np.zeros(n)
    theta_true[np.random.choice(n, NONZEROS, replace=False)] = 5 * np.random.randn(NONZEROS)

    def sigmoid(u):
        return 1 / (1 + np.exp(-u))

    def make_data(num):
        X = np.random.randn(num, n)
        Y = (np.random.rand(num) < sigmoid(X @ theta_true)).astype(float)
        X = np.hstack([X, np.ones((num, 1))])  # constant feature for the intercept
        return X, Y

    X, Y = make_data(m)
    X_test, Y_test = make_data(TEST)
    return X, X_test, Y, Y_test, m, n, np


@app.cell
def _(X, Y, m, n):
    # Form the l1-regularized logistic regression problem.
    import cvxpy as cp

    theta = cp.Variable(n + 1)
    lambd = cp.Parameter(nonneg=True)
    loss = cp.sum(cp.logistic(X @ theta)) - Y @ (X @ theta)
    reg = cp.norm1(theta[:n])  # don't regularize the intercept
    prob = cp.Problem(cp.Minimize(loss / m + lambd * reg))
    return lambd, prob, theta


@app.cell
def _(X, X_test, Y, Y_test, lambd, np, prob, theta):
    def error(X, Y, theta):
        return np.mean((X @ theta > 0) != (Y == 1))

    TRIALS = 30
    lambda_vals = np.logspace(-3, 0, TRIALS)
    train_error = np.zeros(TRIALS)
    test_error = np.zeros(TRIALS)
    for _i, _val in enumerate(lambda_vals):
        lambd.value = _val
        prob.solve()
        train_error[_i] = error(X, Y, theta.value)
        test_error[_i] = error(X_test, Y_test, theta.value)
    return lambda_vals, test_error, train_error


@app.cell
def _(lambda_vals, test_error, train_error):
    # Plot the train and test error over the trade-off curve.
    import matplotlib.pyplot as plt

    plt.plot(lambda_vals, train_error, label="Train error")
    plt.plot(lambda_vals, test_error, label="Test error")
    plt.xscale("log")
    plt.legend(loc="upper left")
    plt.xlabel(r"$\lambda$", fontsize=16)
    plt.ylabel("Error")
    plt.show()
    return


@app.cell
def _():
    import marimo as mo 

    return (mo,)


if __name__ == "__main__":
    app.run()
