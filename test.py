import numpy as np
from NeuralNetwork import loss, activation, layer, network

if __name__ == "__main__":

    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])

    Y = np.array([[0], [1],
                  [1], [0]])

    model = network.Sequential()

    model.add_layer(layer.Dense(n_neurons=3, name='Layer_1', freeze_weights=False, weights=np.array([[0.1, 0.2, 0.3],
                                                                                                     [0.6, 0.4, 0.7]]),
                                bias=np.array([[0, 0, 0]])
                                )
                    )

    model.add_layer(activation.Sigmoid(name='Activation_1'))

    model.add_layer(layer.Dense(n_neurons=1, name='Layer_2', freeze_weights=False, weights=np.array([[0.1],
                                                                                                     [0.4],
                                                                                                     [0.9]]),
                                bias = np.array([[0]])
                                )
                    )

    model.add_layer(activation.Sigmoid(name='Activation_2'))

    model.compile(loss=loss.MSE(), inputs=X, target=Y, batch=3)
    print(model)
    input('Enter to carry on')
    model.train(epochs=5000)

    pred_val = model.predict(inputs=X)
    print(pred_val)