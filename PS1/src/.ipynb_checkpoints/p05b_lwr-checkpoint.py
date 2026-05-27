import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel


def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a LWR model

    model = LocallyWeightedLinearRegression(tau = tau)

    model.fit(x_train, y_train)

    x_eval, y_eval = util.load_dataset(eval_path, add_intercept = True)

    y_pred = model.predict(x_eval)

    # Get MSE value on the validation set

    mse = np.mean((y_pred - y_eval) ** 2)

    print(mse)

    plt.figure()
    plt.plot(x_train, y_train, 'bx', linewidth=2)
    plt.plot(x_eval, y_pred, 'ro', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('output/p05b.png')
    # Plot validation predictions on top of training set
    # No need to save predictions
    # Plot data
    # *** END CODE HERE ***


class LocallyWeightedLinearRegression(LinearModel):
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        self.x = x
        self.y = y
        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape

        self.theta = np.zeros(n)

        for i in range(m) :
        
            W = np.diag(np.exp(-np.sum((self.x - self.x[i]) ** 2, axis=1) / (2 * self.tau ** 2)))

            theta_old = np.copy(self.theta)

            # compute theta = (X^T W X)^(-1) X^T W y
            self.theta = np.linalg.inv(self.x.T @ W @ self.x) @ (self.x.T @ W @ self.y)

        return x @ self.theta
            
        # *** END CODE HERE ***
