import numpy as np

class _init(object):
    """Initialisation of Optimiser
    """

    def _init_optimiser(self, lr, freeze_weights, momentum, nesterov):
        """Initialise optimiser with parameters

        Args:
            freeze_weights (bool): True to not update params and False to update params
            lr (float): Learning rate
        """
        self.lr = lr
        self.freeze_weights = freeze_weights
        self.momentum = momentum
        self.nesterov = nesterov

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

    def __init__(self, lr=1, freeze_weights=False, momentum=0.9, nesterov=False):
        super()._init_optimiser(lr=lr, freeze_weights=freeze_weights,
                                momentum=momentum, nesterov=nesterov)

    def backward(self, activation_prev, weights, bias, upstream_grad):

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
                weights = weights - (self.lr * self.delta_weights_velocity)
                bias = bias - (self.lr * self.delta_bias_velocity)
            elif self.nesterov == True:
                weights = weights - (self.lr * (self.delta_weights_velocity * self.momentum + delta_weights))
                bias = bias - (self.lr * (self.delta_bias_velocity * self.momentum + delta_bias))

        return weights, bias, delta_grad