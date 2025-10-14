import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
from three_layer_neural_network import NeuralNetwork, generate_data, plot_decision_boundary

class DeepNeuralNetwork(NeuralNetwork):
    """
    This class builds and trains a deep neural network with arbitrary number of layers
    """
    def __init__(self, layer_dims, actFun_type='relu', reg_lambda=0.01, seed=0):
        '''
        :param layer_dims: list defining the network structure
        :param actFun_type: type of activation function. 3 options: 'tanh', 'sigmoid', 'relu'
        :param reg_lambda: regularization coefficient
        :param seed: random seed
        '''
        self.layer_dims = layer_dims
        self.num_layers = len(layer_dims) - 1
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda
        
        # initialize weights and biases
        np.random.seed(seed)
        self.weights = []
        self.biases = []
        
        for i in range(self.num_layers):
            dim_prev = layer_dims[i]
            dim_curr = layer_dims[i + 1]
            
            W = np.random.randn(dim_prev, dim_curr) / np.sqrt(dim_prev)
            b = np.zeros((1, dim_curr))
            self.weights.append(W)
            self.biases.append(b)

    def feedforward(self, X, actFun):
        '''
        feedforward implements forward propagation through all layers, and computes the output probabilities
        :param X: input data, shape (num_examples, input_dim)
        :param actFun: activation function
        :return: caches: cached values (z and a) for each layer, used in backpropagation
        '''
        caches = []
        a = X
        
        for i in range(self.num_layers):
            a_prev = a 
            W = self.weights[i]
            b = self.biases[i]
            
            z = a_prev.dot(W) + b
            
            if i == self.num_layers - 1:
                exp_scores = np.exp(z)
                a = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            else:
                a = actFun(z)
            
            caches.append({
                'a_prev': a_prev,
                'z': z
            })
        
        self.probs = a
        return caches

    def calculate_loss(self, X, y):
        '''
        calculate_loss computes the loss for prediction
        :param X: input data, shape (num_examples, input_dim)
        :param y: given labels, shape (num_examples,)
        :return: total loss (data loss + regularization loss)
        '''
        num_examples = len(X)
        
        self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
        
        logprobs = -np.log(self.probs[range(num_examples), y])
        data_loss = np.sum(logprobs)
        
        reg_loss = 0
        for W in self.weights:
            reg_loss += np.sum(np.square(W))
        
        reg_loss = self.reg_lambda / 2 * reg_loss
        total_loss = (1.0 / num_examples) * (data_loss + reg_loss)
        return total_loss

    def backprop(self, probs, caches, X, y):
        '''
        backprop implements backpropagation to compute the gradients used to update the parameters in the backward step
        :param probs: output probabilities from forward propagation, shape (num_examples, output_dim)
        :param caches: cached intermediate values from forward propagation
        :param X: input data, shape (num_examples, input_dim)
        :param y: given labels, shape (num_examples,)
        :return: grads_W: gradients for weight
                 grads_b: gradients for bias
        '''
        num_examples = len(X)
        grads_W = []
        grads_b = []
        
        output_dim = self.layer_dims[-1]
        y_onehot = np.zeros((num_examples, output_dim))
        y_onehot[range(num_examples), y] = 1
        
        dz = probs - y_onehot
        
        for i in reversed(range(self.num_layers)):
            cache = caches[i]
            a_prev = cache['a_prev']
            z = cache['z']
            
            dW = (a_prev.T).dot(dz)
            db = np.sum(dz, axis=0, keepdims=True)
            
            grads_W.insert(0, dW)
            grads_b.insert(0, db)
            
            if i > 0:
                # dz_prev = dz * W^T * diff_actFun(z_prev)
                dz = dz.dot(self.weights[i].T) * self.diff_actFun(caches[i-1]['z'], self.actFun_type)
        
        return grads_W, grads_b

    def fit_model(self, X, y, epochs=20000, epsilon=0.01, print_loss=False):
        '''
        fit_model uses backpropagation to train the network
        :param X: input data, shape (num_examples, input_dim)
        :param y: given labels
        :param epochs: the number of times that the algorithm runs through the whole dataset
        :param epsilon: learning rate
        :param print_loss: print the loss or not
        :return:
        '''
        for epoch in range(epochs):
            # Forward propagation
            caches = self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
            
            # Backpropagation
            grads_W, grads_b = self.backprop(self.probs, caches, X, y)
            
            # Add regularization terms here
            for i in range(self.num_layers):
                grads_W[i] += self.reg_lambda * self.weights[i]
            
            # Gradient descent parameter update
            for i in range(self.num_layers):
                self.weights[i] -= epsilon * grads_W[i]
                self.biases[i] -= epsilon * grads_b[i]
            
            if print_loss and epoch % 1000 == 0:
                loss = self.calculate_loss(X, y)
                print(f"Epoch {epoch}: Loss = {loss:.6f}")

    def save_decision_boundary(self, X, y, filename='decision_boundary.png'):
        '''
        save the decision boundary plot
        :param X: input data
        :param y: given labels
        :param filename: output filename
        :return:
        '''
        # Set min and max values and give it some padding
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        h = 0.01
        # Generate a grid of points with distance h between them
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        # Predict the function value for the whole grid
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        # Plot the contour and training examples
        plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

def main():
    '''
    generate data, build model, train and visualize results
    '''
    # Generate Make-Moons dataset
    X, y = generate_data()
    
    # Build deep neural network model
    # layer_dims: [input_dim, hidden1_dim, hidden2_dim, ..., output_dim]
    # Example: [2, 10, 10, 2]: 2 input nodes, 2 hidden layers with 10 nodes each, 2 output nodes
    
    # baseline
    model = DeepNeuralNetwork(
        layer_dims=[2, 10, 10, 2],
        actFun_type='relu',
        reg_lambda=0.01,
        seed=0
    )
    model.fit_model(X, y, epochs=20000, epsilon=0.01, print_loss=True)
    model.save_decision_boundary(X, y, filename='2x20_relu.png')
    
    # Experiment 1 - act fun
    model = DeepNeuralNetwork(
        layer_dims=[2, 10, 10, 2],
        actFun_type='tanh',
        reg_lambda=0.01,
        seed=0
    )
    model.fit_model(X, y, epochs=20000, epsilon=0.01, print_loss=True)
    model.save_decision_boundary(X, y, filename='2x20_tanh.png')

    model = DeepNeuralNetwork(
        layer_dims=[2, 10, 10, 2],
        actFun_type='sigmoid',
        reg_lambda=0.01,
        seed=0
    )
    model.fit_model(X, y, epochs=20000, epsilon=0.01, print_loss=True)
    model.save_decision_boundary(X, y, filename='2x20_sigmoid.png')
    
    # Experiment 2 - deep / shallow
    model = DeepNeuralNetwork(
        layer_dims=[2, 50, 2],
        actFun_type='relu',
        reg_lambda=0.01,
        seed=0
    )
    model.fit_model(X, y, epochs=20000, epsilon=0.01, print_loss=True)
    model.save_decision_boundary(X, y, filename='1x50_relu.png')
    
    model = DeepNeuralNetwork(
        layer_dims=[2, 8, 8, 8, 2],
        actFun_type='relu',
        reg_lambda=0.01,
        seed=0
    )
    model.fit_model(X, y, epochs=20000, epsilon=0.01, print_loss=True)
    model.save_decision_boundary(X, y, filename='3x8_relu.png')
    
    # Experiment 3 - layer size
    model = DeepNeuralNetwork(
        layer_dims=[2, 3, 3, 2],
        actFun_type='relu',
        reg_lambda=0.01,
        seed=0
    )
    model.fit_model(X, y, epochs=20000, epsilon=0.01, print_loss=True)
    model.save_decision_boundary(X, y, filename='2x3_relu.png')
    
    model = DeepNeuralNetwork(
        layer_dims=[2, 100, 100, 2],
        actFun_type='relu',
        reg_lambda=0.01,
        seed=0
    )
    model.fit_model(X, y, epochs=20000, epsilon=0.01, print_loss=True)
    model.save_decision_boundary(X, y, filename='2x100_relu.png')
    
    # Experiment 4 - reg lambda
    model = DeepNeuralNetwork(
        layer_dims=[2, 100, 100, 2],
        actFun_type='relu',
        reg_lambda=0.0,
        seed=0
    )
    model.fit_model(X, y, epochs=20000, epsilon=0.01, print_loss=True)
    model.save_decision_boundary(X, y, filename='2x100_relu_no_reg.png')
    
    model = DeepNeuralNetwork(
        layer_dims=[2, 100, 100, 2],
        actFun_type='relu',
        reg_lambda=0.5,
        seed=0
    )
    model.fit_model(X, y, epochs=20000, epsilon=0.01, print_loss=True)
    model.save_decision_boundary(X, y, filename='2x100_relu_reg_0.5.png')

if __name__ == "__main__":
    main()