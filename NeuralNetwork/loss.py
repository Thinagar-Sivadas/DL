import numpy as np

class _init(object):
    """Initialise Loss Function
    """

    def _init_loss(self):
        self.name = 'loss'

class MSE(_init):
    """MSE Loss Function
    """

    def __init__(self):
        super()._init_loss()

    def forward(self, target, pred_val):
        """Forward propagation
        https://ml-cheatsheet.readthedocs.io/en/latest/backpropagation.html

        Args:
            target (samples, n_features): Target value
            pred_val (samples, n_features): Predicted value
        """
        self.target = target
        self.pred_val = pred_val
        diff = self.target - self.pred_val
        self.output = np.sum(diff**2) / (self.target.size * 2)

    def backward(self):
        """Backward propagation
        https://ml-cheatsheet.readthedocs.io/en/latest/backpropagation.html

        Returns:
            samples, n_features: Gradient of loss
        """
        diff = self.target - self.pred_val
        self.delta_grad = -1 * diff / self.target.size
        return self.delta_grad

class MAE(_init):
    """MAE Loss Function
    """

    def __init__(self):
        super()._init_loss()

    def forward(self, target, pred_val):
        """Forward propagation
        https://www.programmersought.com/article/95765481689/

        Args:
            target (samples, n_features): Target value
            pred_val (samples, n_features): Predicted value
        """
        self.target = target
        self.pred_val = pred_val
        diff = np.abs(self.target - self.pred_val)
        self.output = np.sum(diff) / self.target.size

    def backward(self):
        """Backward propagation
        https://www.programmersought.com/article/95765481689/

        Returns:
            samples, n_features: Gradient of loss
        """
        diff = self.target - self.pred_val
        self.delta_grad = np.where(diff > 0, -1, 1) / self.target.size
        return self.delta_grad

class Hubber(_init):
    """Hubber Loss Function
    """

    def __init__(self, delta=1):
        """
        Args:
            delta (int, optional): When delta tends to inf, it functions like MSE
            and when delta tends to -inf, functions like MAE. Defaults to 1.
        """
        super()._init_loss()
        self.delta = delta

    def forward(self, target, pred_val):
        """Forward propagation
        https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html#huber

        Args:
            target (samples, n_features): Target value
            pred_val (samples, n_features): Predicted value
        """
        self.target = target
        self.pred_val = pred_val
        diff = self.target - self.pred_val
        mask = np.where(np.abs(diff) < self.delta,
                        diff**2 / 2,
                        self.delta * (np.abs(diff) - 0.5 * self.delta)
                        )
        self.output = mask.sum() / self.target.size

    def backward(self):
        """Backward propagation
        https://ml-cheatsheet.readthedocs.io/en/latest/backpropagation.html
        https://www.programmersought.com/article/95765481689/

        Returns:
            samples, n_features: Gradient of loss
        """
        diff = self.target - self.pred_val
        self.delta_grad = np.where(np.abs(diff) < self.delta,
                                   -1 * diff,
                                   np.where(diff > 0, -1 * self.delta, 1 * self.delta)
                                   ) / self.target.size
        return self.delta_grad

class LogCosh(_init):
    """LogCosh Loss Function
    """

    def __init__(self):
        super()._init_loss()

    def forward(self, target, pred_val):
        """Forward propagation
        https://github.com/tuantle/regression-losses-pytorch

        Args:
            target (samples, n_features): Target value
            pred_val (samples, n_features): Predicted value
        """
        self.target = target
        self.pred_val = pred_val
        diff = self.target - self.pred_val
        self.output = np.sum(np.log((np.exp(diff) + np.exp(-diff)) / 2)
                             ) / self.target.size

    def backward(self):
        """Backward propagation
        https://en.wikipedia.org/wiki/Hyperbolic_functions
        https://github.com/tuantle/regression-losses-pytorch

        Returns:
            samples, n_features: Gradient of loss
        """
        diff = self.target - self.pred_val
        self.delta_grad = -1 * ((np.exp(diff) - np.exp(-diff)) / (np.exp(diff) + np.exp(-diff))
                                ) / self.target.size
        return self.delta_grad

class Quantile(_init):
    """Quantile Loss Function
    """

    def __init__(self, quantile=0.9):
        """
        Args:
            quantile (float, optional): Quantile value determines the models predicition
            intervals. For example, fitting a model with quantile 0.95 and (1-0.95)=0.05
            is described as a 90% prediction interval(0.95-0.05). Defaults to 0.9.
        https://www.evergreeninnovations.co/blog-quantile-loss-function-for-machine-learning/
        http://ethen8181.github.io/machine-learning/ab_tests/quantile_regression/quantile_regression.html
        https://stats.stackexchange.com/questions/154677
        """
        super()._init_loss()
        self.quantile = quantile

    def forward(self, target, pred_val):
        """Forward propagation
        http://ethen8181.github.io/machine-learning/ab_tests/quantile_regression/quantile_regression.html

        Args:
            target (samples, n_features): Target value
            pred_val (samples, n_features): Predicted value
        """
        self.target = target
        self.pred_val = pred_val
        diff = self.target - self.pred_val
        mask = np.maximum(self.quantile * diff, (self.quantile - 1) * diff)
        self.output = mask.sum() / self.target.size

    def backward(self):
        """Backward propagation
        http://ethen8181.github.io/machine-learning/ab_tests/quantile_regression/quantile_regression.html

        Returns:
            samples, n_features: Gradient of loss
        """
        diff = self.target - self.pred_val
        self.delta_grad = np.where(diff >= 0,
                                   -self.quantile,
                                   1 - self.quantile,
                                   ) / self.target.size
        return self.delta_grad

class CrossEntropy(_init):
    """CrossEntropy Loss Function
    """

    def __init__(self):
        super()._init_loss()

    def forward(self, target, pred_val):
        """Forward propagation
        https://math.stackexchange.com/questions/2503428
        https://gombru.github.io/2018/05/23/cross_entropy_loss/

        Args:
            target (samples, n_features): Target value, features must be either 0 or 1
            pred_val (samples, n_features): Predicted value
        """
        self.target = target
        self.pred_val = pred_val
        diff = -self.target * (np.log(self.pred_val)) - (1 - self.target) * np.log(1 - self.pred_val)
        self.output = diff.sum() / self.target.size

    def backward(self):
        """Backward propagation
        https://math.stackexchange.com/questions/2503428
        https://gombru.github.io/2018/05/23/cross_entropy_loss/

        Returns:
            samples, n_features: Gradient of loss
        """
        self.delta_grad = ((-self.target / self.pred_val)
                           + ((1 - self.target) / (1 - self.pred_val))
                           ) / self.target.size
        return self.delta_grad

class SquaredHinge(_init):
    """SquaredHinge Loss Function
    """

    def __init__(self):
        super()._init_loss()

    def forward(self, target, pred_val):
        """Forward propagation
        https://www.quora.com/Why-is-squared-hinge-loss-differentiable
        https://stackoverflow.com/questions/56864276

        Args:
            target (samples, n_features): Target value, features must be either -1 or 1
            pred_val (samples, n_features): Predicted value
        """
        self.target = target
        self.pred_val = pred_val
        diff = 1 - (self.target * self.pred_val)
        self.output = np.maximum(0, np.power(diff, 2)).sum() / (self.target.size * 2)

    def backward(self):
        """Backward propagation
        https://www.quora.com/Why-is-squared-hinge-loss-differentiable
        https://stackoverflow.com/questions/56864276

        Returns:
            samples, n_features: Gradient of loss
        """
        diff = 1 - (self.target * self.pred_val)
        self.delta_grad = np.where(diff > 0, diff * -self.target, 0) / self.target.size
        return self.delta_grad