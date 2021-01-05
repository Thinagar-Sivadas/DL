import numpy as np

class _init(object):
    """Intialisation of Layers
    """

    def _init_layer(self, name, n_neurons, weights, bias, freeze_weights, lr, optimiser):
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
        self.optimiser = optimiser

        if self.weights is not None and self.bias is not None:
            self.params = self.weights.size + self.bias.size
            if self.weights.shape[1] != self.n_neurons or self.bias.shape[1] != self.n_neurons:
                raise ValueError(f'n_neurons chosen for {self.name} does not tally with initialised weights and bias')
        else:
            self.params = None

        if not hasattr(self.optimiser, '__module__') or self.optimiser.__module__ not in 'NeuralNetwork.optimiser':
            raise ValueError(f'Selected optimiser for {self.name} is invalid')

    def _init_param(self, n_inputs, next_layer):
        """Intialise weights and bias
        https://towardsdatascience.com/26c649eb3b78
        https://towardsdatascience.com/954fb9b47c79
        https://www.youtube.com/watch?v=yWCj95DdWXs&ab_channel=NPTEL-NOCIITM
        https://www.youtube.com/watch?v=s2coXdufOzE&t=211s&ab_channel=Deeplearning.ai

        Args:
            n_inputs (int): Number of input neurons to current layer
            next_layer (obj): Contains object of next layer
        """
        self.n_inputs = n_inputs

        if next_layer.__class__.__name__ in ['ReLU', 'LeakyReLU']:
            # Initialisation for ReLU, LeakyReLU
            self.weights = np.random.randn(self.n_inputs, self.n_neurons) * np.sqrt(2 / self.n_inputs)
        else:
            # Initialisation for any others
            self.weights = np.random.randn(self.n_inputs, self.n_neurons) * np.sqrt(1 / self.n_inputs)

        self.bias = np.zeros(shape=(1, self.n_neurons))
        self.params = self.weights.size + self.bias.size

class Dense(_init):
    """Dense Layer
    """

    def __init__(self, name, n_neurons, optimiser, weights=None, bias=None, freeze_weights=False, lr=1):
        super()._init_layer(name=name, n_neurons=n_neurons, weights=weights,
                           bias=bias, freeze_weights=freeze_weights, lr=lr, optimiser=optimiser)

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
        self.delta_weights, self.delta_bias, self.delta_grad = self.optimiser.backward(
                                                                activation_prev=self.activation_prev,
                                                                weights=self.weights,
                                                                upstream_grad=upstream_grad)
        if self.freeze_weights == False:
            self._update_params()

    def _update_params(self):
        self.weights = self.weights - (self.lr * self.delta_weights)
        self.bias = self.bias - (self.lr * self.delta_bias)