import numpy as np

class L2Reg(object):
    """L2 Regularisation
    """

    def __init__(self, alpha):
        self.alpha = alpha

    def forward(self, weights, batch_size):
        """Forward propagation
        https://www.youtube.com/watch?v=6g0t3Phly2M&ab_channel=DeepLearningAI
        https://www.youtube.com/watch?v=lg4OLAjxRcQ&t=1237s&ab_channel=NPTEL-NOCIITM

        Args:
            weights (array): Weights matrix
            batch_size (int): Batch size
        """
        self.weights = weights
        self.batch_size = batch_size
        self.output = (self.alpha * (self.weights ** 2).sum()) / (self.batch_size * 2)

    def backward(self):
        """Backward propagation

        Returns:
            weights (array): Gradient of weight matrix
        """
        self.delta_grad = (self.alpha * self.weights) / self.batch_size
        return self.delta_grad