import numpy as np

class Initialise(object):
    """Parameter Intialisation
    """

    def __init__(self, weight_init):
        weight_init_list = ['Kaiming', 'He', 'Glorot', 'Xavier', None]
        self.weight_init = weight_init
        if self.weight_init not in weight_init_list:
            raise ValueError(f'Not in list of weights intialisation {weight_init_list}')

    def _init_param(self, n_input_neurons, next_layer):
        """Intialise weights and bias
        https://towardsdatascience.com/26c649eb3b78
        https://towardsdatascience.com/954fb9b47c79
        https://www.youtube.com/watch?v=yWCj95DdWXs&ab_channel=NPTEL-NOCIITM
        https://www.youtube.com/watch?v=s2coXdufOzE&t=211s&ab_channel=Deeplearning.ai

        Args:
            n_input_neurons (int): Number of input neurons to current layer
            next_layer (obj): Contains object of next layer
        """
        if self.weight_init in ['Kaiming', 'He'] or next_layer.__class__.__name__ in ['ReLU', 'LeakyReLU']:
            # Initialisation for ReLU, LeakyReLU, Kaiming/He Initialisation
            print('Kaiming/He Weights Initialisation\n')
            self.weights = np.random.randn(n_input_neurons, self.n_neurons) * np.sqrt(2 / n_input_neurons)
        else:
            # Initialisation for any others, Xavier/Glorot initialisation
            print('Xavier/Glorot Weights Initialisation\n')
            self.weights = np.random.randn(n_input_neurons, self.n_neurons) * np.sqrt(1 / n_input_neurons)

        self.bias = np.zeros(shape=(1, self.n_neurons))
