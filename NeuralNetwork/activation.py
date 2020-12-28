import numpy as np

class _init(object):
    """Initialise Activation Function
    """

    def _init_activation(self, name):
        self.name = name

class ReLU(_init):
    """ReLU Activation Function
    """

    def __init__(self, name):
        """
        Args:
            name (str): Initialise activation class with name
        """
        super()._init_activation(name=name)

    def forward(self, inputs):
        """Forward propagation
        https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#relu

        Args:
            inputs (samples, n_neurons): Performs activation
        """
        self.output = np.maximum(0, inputs)

    def backward(self, upstream_grad):
        """Backward propagation
        https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#relu

        Args:
            upstream_grad (samples, n_neurons): Elementwise multiplication with upstream gradient
        """
        self.delta_grad =  upstream_grad * np.where(self.output > 0, 1, self.output)

class LeakyReLU(_init):
    """LeakyReLU Activation Function
    """

    def __init__(self, name, alpha=0.01):
        """
        Args:
            name (str): Initialise activation class with name
            alpha (float, optional): Value must be smaller than 0. Defaults to 0.01.
        """
        super()._init_activation(name=name)
        self.alpha = alpha

    def forward(self, inputs):
        """Forward propagation
        https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#leakyrelu

        Args:
            inputs (samples, n_neurons): Performs activation
        """
        self.output = np.maximum(self.alpha * inputs, inputs)

    def backward(self, upstream_grad):
        """Backward propagation
        https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#leakyrelu

        Args:
            upstream_grad (samples, n_neurons): Elementwise multiplication with upstream gradient
        """
        self.delta_grad =  upstream_grad * np.where(self.output > 0, 1, self.alpha)

class Tanh(_init):
    """Tanh Activation Function
    """

    def __init__(self, name):
        """
        Args:
            name (str): Initialise activation class with name
        """
        super()._init_activation(name=name)

    def forward(self, inputs):
        """Forward propagation
        https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#tanh

        Args:
            inputs (samples, n_neurons): Performs activation
        """
        self.output = (np.exp(inputs) - np.exp(-inputs)) / (np.exp(inputs) + np.exp(-inputs))

    def backward(self, upstream_grad):
        """Backward propagation
        https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#tanh

        Args:
            upstream_grad (samples, n_neurons): Elementwise multiplication with upstream gradient
        """
        self.delta_grad =  upstream_grad * (1 - np.power(self.output, 2))

class Softmax(_init):
    """Softmax Activation Function
    """

    def __init__(self, name):
        """
        Args:
            name (str): Initialise activation class with name
        """
        super()._init_activation(name=name)

    def forward(self, inputs):
        """Forward propagation
        https://mattpetersen.github.io/softmax-with-cross-entropy
        https://math.stackexchange.com/questions/2843505/derivative-of-softmax-without-cross-entropy

        Args:
            inputs (samples, n_neurons): Performs activation
        """
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def backward(self, upstream_grad):
        """Backward propagation
        https://mattpetersen.github.io/softmax-with-cross-entropy
        https://math.stackexchange.com/questions/2843505/derivative-of-softmax-without-cross-entropy

        Args:
            upstream_grad (samples, n_neurons): Dot product with upstream gradient
        """
        jacobian_matrix = np.array([(np.diagflat(row) - np.dot(row.reshape(1,-1).T, row.reshape(1,-1)))
                                    for row in self.output])
        self.delta_grad = np.concatenate([np.dot(ind_upstream_grad.reshape(1,-1), ind_jacobian_matrix)
                                          for ind_upstream_grad, ind_jacobian_matrix in
                                          zip(upstream_grad, jacobian_matrix)],
                                         axis=0)

class Sigmoid(_init):
    """Sigmoid Activation Function
    """

    def __init__(self, name):
        """
        Args:
            name (str): Initialise activation class with name
        """
        super()._init_activation(name=name)

    def forward(self, inputs):
        """Forward propagation
        https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#sigmoid

        Args:
            inputs (samples, n_neurons): Performs activation
        """
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, upstream_grad):
        """Backward propagation
        https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#sigmoid

        Args:
            upstream_grad (samples, n_neurons): Elementwise multiplication with upstream gradient
        """
        self.delta_grad =  upstream_grad * (self.output * (1 - self.output))