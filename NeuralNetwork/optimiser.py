import numpy as np

class _init(object):
    """Initialisation of Optimiser
    """

    def _init_optimiser(self, lr, freeze_weights, regulariser):
        """Initialise optimiser with parameters

        Args:
            freeze_weights (bool): True to not update params and False to update params
            lr (float): Learning rate
        """
        self.lr = lr
        self.freeze_weights = freeze_weights
        self.regulariser = regulariser
        if not hasattr(self.regulariser, '__module__') or 'regulariser' not in self.regulariser.__module__:
            raise ValueError(f'Selected regulariser is invalid')

class SGD(_init):
    """Stochastic Gradient Descent
    https://www.youtube.com/watch?v=uJryes5Vk1o&ab_channel=Deeplearning.ai
    https://aerinykim.medium.com/73c7368644fa

    Stochastic Gradient Descent with Momentum
    Using the purple version in andrew ng video
    https://www.youtube.com/watch?app=desktop&v=k8fTYJPd3_I&ab_channel=Deeplearning.ai
    https://www.youtube.com/watch?v=ewN0vFYFJ7A&ab_channel=NPTEL-NOCIITM
    https://towardsdatascience.com/stochastic-gradient-descent-with-momentum-a84097641a5d
    https://medium.com/optimization-algorithms-for-deep-neural-networks/gradient-descent-with-momentum-dce805cd8de8

    Stochastic Gradient Descent with Momentum and Nesterov
    https://stats.stackexchange.com/questions/179915/
    https://cs231n.github.io/neural-networks-3/#sgd
    https://www.youtube.com/watch?v=sV9aiEsXanE&t=303s&ab_channel=NPTEL-NOCIITM
    """

    def __init__(self, regulariser, lr=1, freeze_weights=False, momentum=0.9, nesterov=False):
        super()._init_optimiser(lr=lr, freeze_weights=freeze_weights,
                                regulariser=regulariser)
        self.momentum = momentum
        self.nesterov = nesterov

    def backward(self, activation_prev, weights, bias, upstream_grad):

        self.regulariser.forward(weights=weights, batch_size=activation_prev.shape[0])

        delta_weights = np.dot(activation_prev.T, upstream_grad)
        if hasattr(self, 'delta_weights_velocity') == False:
            self.delta_weights_velocity = np.zeros_like(delta_weights)
        self.delta_weights_velocity = (self.momentum * self.delta_weights_velocity) + delta_weights

        delta_bias = np.sum(upstream_grad, axis=0, keepdims=True)
        if hasattr(self, 'delta_bias_velocity') == False:
            self.delta_bias_velocity = np.zeros_like(delta_bias)
        self.delta_bias_velocity = (self.momentum * self.delta_bias_velocity) + delta_bias

        delta_grad = np.dot(upstream_grad, weights.T)

        if self.freeze_weights == False:
            if self.nesterov == False:
                weights = weights - (self.lr * self.delta_weights_velocity) \
                                  - (self.lr * self.regulariser.backward())
                bias = bias - (self.lr * self.delta_bias_velocity)
            elif self.nesterov == True:
                weights = weights - (self.lr * (self.delta_weights_velocity * self.momentum + delta_weights)) \
                                  - (self.lr * self.regulariser.backward())
                bias = bias - (self.lr * (self.delta_bias_velocity * self.momentum + delta_bias))

        return weights, bias, delta_grad

class AdaGrad(_init):
    """AdaGrad
    https://www.youtube.com/watch?v=FKCV76N9Ys0&ab_channel=NPTEL-NOCIITM
    """

    def __init__(self, regulariser, lr=1, freeze_weights=False, eps=1e-10):
        super()._init_optimiser(lr=lr, freeze_weights=freeze_weights,
                                regulariser=regulariser)
        self.eps = eps

    def backward(self, activation_prev, weights, bias, upstream_grad):

        self.regulariser.forward(weights=weights, batch_size=activation_prev.shape[0])

        delta_weights = np.dot(activation_prev.T, upstream_grad)
        if hasattr(self, 'delta_weights_velocity') == False:
            self.delta_weights_velocity = np.zeros_like(delta_weights)
        self.delta_weights_velocity += (delta_weights ** 2)

        delta_bias = np.sum(upstream_grad, axis=0, keepdims=True)
        if hasattr(self, 'delta_bias_velocity') == False:
            self.delta_bias_velocity = np.zeros_like(delta_bias)
        self.delta_bias_velocity += (delta_bias ** 2)

        delta_grad = np.dot(upstream_grad, weights.T)

        if self.freeze_weights == False:
            weights = weights - (self.lr / np.sqrt(self.delta_weights_velocity + self.eps)) * delta_weights \
                                - (self.lr * self.regulariser.backward())
            bias = bias - (self.lr / np.sqrt(self.delta_bias_velocity + self.eps)) * delta_bias

        return weights, bias, delta_grad

class RmsProp(_init):
    """RMSProp
    https://www.youtube.com/watch?v=FKCV76N9Ys0&ab_channel=NPTEL-NOCIITM
    """

    def __init__(self, regulariser, lr=1, freeze_weights=False, eps=1e-10, momentum=0.9):
        super()._init_optimiser(lr=lr, freeze_weights=freeze_weights,
                                regulariser=regulariser)
        self.momentum = momentum
        self.eps = eps

    def backward(self, activation_prev, weights, bias, upstream_grad):

        self.regulariser.forward(weights=weights, batch_size=activation_prev.shape[0])

        delta_weights = np.dot(activation_prev.T, upstream_grad)
        if hasattr(self, 'delta_weights_velocity') == False:
            self.delta_weights_velocity = np.zeros_like(delta_weights)
        self.delta_weights_velocity = (self.momentum * self.delta_weights_velocity) + \
                                      (1 - self.momentum) * (delta_weights ** 2)

        delta_bias = np.sum(upstream_grad, axis=0, keepdims=True)
        if hasattr(self, 'delta_bias_velocity') == False:
            self.delta_bias_velocity = np.zeros_like(delta_bias)
        self.delta_bias_velocity = (self.momentum * self.delta_bias_velocity) + (1 - self.momentum) * (delta_bias ** 2)

        delta_grad = np.dot(upstream_grad, weights.T)

        if self.freeze_weights == False:
            weights = weights - (self.lr / np.sqrt(self.delta_weights_velocity + self.eps)) * delta_weights \
                                - (self.lr * self.regulariser.backward())
            bias = bias - (self.lr / np.sqrt(self.delta_bias_velocity + self.eps)) * delta_bias

        return weights, bias, delta_grad