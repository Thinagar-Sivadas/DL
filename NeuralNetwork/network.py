import numpy as np
import pickle
import plotly.graph_objects as go

class Sequential(object):
    """Sequential Neural Network Modelling
    """

    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def compile(self, loss, inputs, target, batch=8):
        self._check(inputs, target)
        self.loss = loss
        self.inputs = inputs
        self.target = target
        self.batch = batch
        self._compile_forward(inputs=self.inputs)

    def train(self, epochs=100):
        self.cost = []

        epoch = 1
        while epoch < epochs+1:

            batch_inputs, batch_target = self._mini_batch()

            for itr in range(len(batch_inputs)):

                pred_val = self._forward(inputs=batch_inputs[itr], layer_name=None)
                self.loss.forward(target=batch_target[itr], pred_val=pred_val)

                delta_grad = self.loss.backward()
                self._backward(delta_grad=delta_grad)

            cost = float(f'{self.loss.output:.6f}')

            self.cost.append(cost)
            if (epoch % 20) == 0:
                print(f'Epoch:{epoch}/{epochs} ------------------------ Loss:{cost}')
            epoch += 1

        print('\n' + f'-' * 20 + 'Training Completed' + f'-' * 20 + '\n')
        fig = go.Figure()
        fig.add_traces(go.Scatter(x=list(range(1, len(self.cost)+1)),
                                  y=self.cost, mode='lines+markers',
                                  name='Cost'))
        fig.update_layout(title='Cost Function',
                          xaxis_title='Epochs', yaxis_title='Cost',
                          xaxis_tickformat='d')
        fig.show()

    def predict(self, inputs, layer_name=None):
        layer_names = [layer.name for layer in self.layers]
        if layer_name is None:
            return self._forward(inputs=inputs, layer_name=None)
        elif layer_name in layer_names:
            return self._forward(inputs=inputs, layer_name=layer_name)
        else:
            raise ValueError(f'Layer does not exist\n'
                             f'Available layers: {layer_names}')

    def _forward(self, inputs, layer_name=None):
        input_data = inputs
        for layer in self.layers:
            layer.forward(input_data)
            input_data = layer.output
            if layer_name == layer.name:
                break
        return input_data

    def _compile_forward(self, inputs):
        input_neuron = inputs.shape[1]
        number_layers = len(self.layers)
        for ind, layer in enumerate(self.layers):
            if hasattr(layer, 'weights'):

                if ind < number_layers - 1:
                    next_layer = self.layers[ind+1]
                else:
                    next_layer = None

                if layer.weights is None or layer.bias is None:
                    print(f'Initialising weights and bias for {layer.name}')
                    layer._init_param(input_neuron, next_layer=next_layer)
                elif layer.weights.shape[0] != input_neuron:
                    print(f'Reinitialising weights and bias for {layer.name} due to change in input neurons')
                    layer._init_param(input_neuron, next_layer=next_layer)

                input_neuron = layer.n_neurons

    def _backward(self, delta_grad):
        upstream_grad = delta_grad
        for layer in self.layers[::-1]:
            layer.backward(upstream_grad)
            upstream_grad = layer.delta_grad

    def _check(self, inputs, target):
        assert inputs.ndim == target.ndim, (f'Input and target dimension not the same')

        assert inputs.shape[0] == target.shape[0], (f"Input sample size not the "
                                                    f"same as target sample size "
                                                    f"Format (samples, n_features)")
        for layer in self.layers[::-1]:
            if hasattr(layer, 'n_neurons'):
                assert layer.n_neurons == target.shape[1], (f"Output target feature not the "
                                                            f"same as last layer n_neurons "
                                                            f"Format (samples, n_neurons)")
                break

    def _mini_batch(self):
        randomize = np.arange(len(self.inputs))
        np.random.shuffle(randomize)
        self.inputs = self.inputs[randomize]
        self.target = self.target[randomize]
        batch_inputs = []
        batch_target = []
        for ind in range(0, len(self.inputs), self.batch):
            batch_inputs.append(self.inputs[ind:ind+self.batch])
            batch_target.append(self.target[ind:ind+self.batch])

        return batch_inputs, batch_target

    def save_weights(self):
        with open('model.pkl', 'wb') as output:
            pickle.dump(self.layers, output, pickle.HIGHEST_PROTOCOL)

    def load_weights(self):
        with open('model.pkl', 'rb') as output:
            self.layers = pickle.load(output)

    def __str__(self):
        header = ['Layer_Name', 'Type', 'Output Shape', 'Param']
        layer_names = []
        layer_types = []
        layer_module = []
        layer_shape = []
        param = []
        max_length = len(max(header, key=len))
        total_param = 0
        non_trainable_param = 0
        var = ''

        for layer in self.layers:
            layer_names.append(layer.name)
            layer_types.append(layer.__class__.__name__)
            layer_module.append(layer.__module__)

            length = max(len(layer_names[-1]), len(layer_types[-1]))
            if length > max_length:
                max_length = length

            if 'layer' in layer_module[-1]:
                layer_shape.append((None, layer.n_neurons))
                param.append(layer.params)
                if layer.params is not None:
                    total_param += param[-1]
                    if layer.optimiser.freeze_weights:
                        non_trainable_param += param[-1]
            else:
                layer_shape.append(layer_shape[-1])
                param.append(None)

        max_length += 10

        var += '-' * (max_length * len(header)) + '\n'
        for val in header:
            var += val + ' ' * (max_length - len(val))
        var += '\n'
        var += '=' * max_length * len(header) + '\n'

        for ind, name in enumerate(layer_names):
            var += name + ' ' * (max_length - len(name))
            var += layer_types[ind] + ' ' * (max_length - len(layer_types[ind]))
            var += str(layer_shape[ind]) + ' ' * (max_length - len(str(layer_shape[ind])))
            var += str(param[ind]) + ' ' * (max_length - len(str(param[ind])))
            var += '\n'
            var += '=' * (max_length * len(header)) + '\n'

        var += f'Total Params: {total_param}\n'
        var += f'Trainable params:: {total_param - non_trainable_param}\n'
        var += f'Non-Trainable params: {non_trainable_param}\n'

        return var

