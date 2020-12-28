import numpy as np

class _init(object):
    """Intialisation of Layers
    """

    def _init_layer(self, name, n_neurons, weights, bias, freeze_weights, lr):
        """Intialise layer with parameters

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
        self.n_neurons = n_neurons
        self.weights = weights
        self.bias = bias
        self.lr = lr
        self.freeze_weights = freeze_weights

        if self.weights is not None and self.bias is not None:
            self.params = self.weights.size + self.bias.size
            if self.weights.shape[1] != self.n_neurons or self.bias.shape[1] != self.n_neurons:
                raise ValueError(f'n_neurons chosen for {self.name} does not tally with initialised weights and bias')
        else:
            self.params = None

class Dense(_init):
    """Dense Layer
    """

    def __init__(self, name, n_neurons, weights=None, bias=None, freeze_weights=False, lr=1):
        super()._init_layer(name=name, n_neurons=n_neurons, weights=weights,
                           bias=bias, freeze_weights=freeze_weights, lr=lr)

    def _init_param(self, n_inputs):
        self.n_inputs = n_inputs
        self.weights = 0.10 * np.random.randn(self.n_inputs, self.n_neurons)
        self.bias = np.zeros(shape=(1, self.n_neurons))
        self.params = self.weights.size + self.bias.size

    def forward(self, inputs):
        """Forward propagation
        https://www.kdnuggets.com/2019/08/numpy-neural-networks-computational-graphs.html

        Args:
            inputs (samples, n_neurons): [description]
        """
        if self.weights is None or self.bias is None:
            print(f'Initialising weights and bias for {self.name}')
            self._init_param(inputs.shape[1])
        elif self.weights.shape[0] != inputs.shape[1]:
            print(f'Reinitialising weights and bias for {self.name} due to change in input neurons')
            self._init_param(inputs.shape[1])
        self.activation_prev = inputs
        self.output = np.dot(self.activation_prev, self.weights) + self.bias

    def backward(self, upstream_grad):
        """Backward propagation
        https://www.kdnuggets.com/2019/08/numpy-neural-networks-computational-graphs.html

        Args:
            upstream_grad (samples, n_neurons): Dot product with upstream gradient
        """
        self.delta_weights = np.dot(self.activation_prev.T, upstream_grad)
        self.delta_bias = np.sum(upstream_grad, axis=0, keepdims=True)
        self.delta_grad = np.dot(upstream_grad, self.weights.T)
        if self.freeze_weights == False:
            self._update_params()

    def _update_params(self):
        self.weights = self.weights - (self.lr * self.delta_weights)
        self.bias = self.bias - (self.lr * self.delta_bias)