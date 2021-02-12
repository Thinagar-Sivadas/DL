import numpy as np
from NeuralNetwork.initialiser import Initialise

class _init(object):
    """Intialisation of Layers
    """

    def _init_layer(self, name, n_neurons, weights, bias, optimiser):
        """Initialise layer with parameters

        Args:
            name (str): Layer name
            n_neurons (int): Number of neurons in the layer
            weights (prev_layer_n_neurons, n_neurons): Weight matrix
            bias (1, n_neurons): Bias matrix
            freeze_weights (bool): True to not update params and False to update params
            lr (float): Learning rate

        Raises:
            ValueError: If chosen n_neurons and self initialised weights and bias matrix does not tally
        """
        self.name = name
        if not isinstance(self.name, str):
            raise ValueError('Must be a string')
        self.n_neurons = n_neurons
        self.weights = weights
        self.bias = bias
        self.optimiser = optimiser

        if not hasattr(self.optimiser, '__module__') or 'optimiser' not in self.optimiser.__module__:
            raise ValueError(f'Selected optimiser for {self.name} is invalid')

        if (self.weights.size != 0) and (self.bias.size != 0) and \
           ((self.weights.shape[1] != self.n_neurons) or (self.bias.shape[1] != self.n_neurons)):
               raise ValueError(f'n_neurons chosen for {self.name} does not tally with initialised weights and bias')

class Dense(_init, Initialise):
    """Dense Layer
    """

    def __init__(self, name, n_neurons, optimiser, weight_init=None,
                 weights=np.empty(shape=(0,0)), bias=np.empty(shape=(0,0))):

        _init._init_layer(self, name=name, n_neurons=n_neurons, weights=weights,
                          bias=bias, optimiser=optimiser)
        Initialise.__init__(self, weight_init=weight_init)

    def forward(self, inputs):
        """Forward propagation
        https://www.kdnuggets.com/2019/08/numpy-neural-networks-computational-graphs.html

        Args:
            inputs (samples, n_neurons): Dot product with input
        """
        self.activation_prev = inputs
        self.output = np.dot(self.activation_prev, self.weights) + self.bias

    def backward(self, upstream_grad):
        """Backward propagation
        https://www.kdnuggets.com/2019/08/numpy-neural-networks-computational-graphs.html

        Args:
            upstream_grad (samples, n_neurons): Dot product with upstream gradient
        """
        self.weights, self.bias, self.delta_grad = self.optimiser.backward(
                                                                activation_prev=self.activation_prev,
                                                                weights=self.weights,
                                                                bias=self.bias,
                                                                upstream_grad=upstream_grad)