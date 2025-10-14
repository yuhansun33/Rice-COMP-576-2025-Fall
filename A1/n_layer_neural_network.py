import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
from three_layer_neural_network import NeuralNetwork, generate_data
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix


def generate_cancer_data():
    '''
    generate data
    :return: X: input data, y: given labels
    '''
    # Scikit-learn 8.1.6. Breast cancer Wisconsin (diagnostic) dataset
    cancer = datasets.load_breast_cancer()
    X, y = cancer.data, cancer.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X.shape[1], X_train, X_test, y_train, y_test

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
                # Numerical stable softmax: subtract max to prevent overflow
                # refer to https://blester125.com/blog/softmax.html
                z_max = np.max(z, axis=1, keepdims=True)
                exp_scores = np.exp(z - z_max)
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

def train_moon(X, y, layer_dims, actFun_type='relu', reg_lambda=0.01, seed=0, 
                   epochs=20000, epsilon=0.01, print_loss=True, filename='output.png'):
    '''
    Train a deep neural network and save the decision boundary plot
    :param X: input data
    :param y: given labels
    :param layer_dims: list defining the network structure
    :param actFun_type: activation function type
    :param reg_lambda: regularization coefficient
    :param seed: random seed
    :param epochs: number of training iterations
    :param epsilon: learning rate
    :param print_loss: whether to print loss during training
    :param filename: output filename for the decision boundary plot
    :return:
    '''
    model = DeepNeuralNetwork(
        layer_dims=layer_dims,
        actFun_type=actFun_type,
        reg_lambda=reg_lambda,
        seed=seed
    )
    model.fit_model(X, y, epochs=epochs, epsilon=epsilon, print_loss=print_loss)
    model.save_decision_boundary(X, y, filename=filename)
    return

def train_cancer(X_train, y_train, X_test, y_test, layer_dims, 
                      actFun_type='tanh', reg_lambda=0.01, seed=0,
                      epochs=20000, epsilon=0.01, print_loss=True, name='model'):
    '''
    Train a deep neural network and evaluate on test set
    :param X_train: training data
    :param y_train: training labels
    :param X_test: test data
    :param y_test: test labels
    :param layer_dims: list defining the network structure
    :param actFun_type: activation function type
    :param reg_lambda: regularization coefficient
    :param seed: random seed
    :param epochs: number of training iterations
    :param epsilon: learning rate
    :param print_loss: whether to print loss during training
    :param name: name of the experiment for printing
    :return:
    '''
    model = DeepNeuralNetwork(
        layer_dims=layer_dims,
        actFun_type=actFun_type,
        reg_lambda=reg_lambda,
        seed=seed
    )
    model.fit_model(X_train, y_train, epochs=epochs, epsilon=epsilon, print_loss=print_loss)
    y_pred = model.predict(X_test)
    
    result_text = f"\n{name}\n"
    result_text += f"Confusion matrix:\n{confusion_matrix(y_test, y_pred)}\n"
    result_text += f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n"
    
    print(result_text)
    # Append to file
    with open("cancer_results.txt", 'a') as f:
        f.write(result_text)
    
    return

def main():
    '''
    generate data, build model, train and visualize results
    '''
    # ##########################################################
    # Make-Moons dataset
    # ##########################################################
    
    # Generate Make-Moons dataset
    X, y = generate_data()
    
    # Build deep neural network model
    # layer_dims: [input_dim, hidden1_dim, hidden2_dim, ..., output_dim]
    # Example: [2, 10, 10, 2]: 2 input nodes, 2 hidden layers with 10 nodes each, 2 output nodes
    
    # baseline
    train_moon(
        X, y, 
        [2, 10, 10, 2], 
        actFun_type='relu',
        filename='n_layers_images/2x10_relu.png'
    )
    
    # Experiment 1 - act func
    train_moon(
        X, y, 
        [2, 10, 10, 2], 
        actFun_type='tanh', 
        filename='n_layers_images/2x10_tanh.png'
    )
    train_moon(
        X, y, 
        [2, 10, 10, 2], 
        actFun_type='sigmoid', 
        filename='n_layers_images/2x10_sigmoid.png')
    
    # Experiment 2 - deep / shallow
    train_moon(
        X, y, 
        [2, 50, 2], 
        filename='n_layers_images/1x50_relu.png')
    train_moon(
        X, y, 
        [2, 8, 8, 8, 2], 
        filename='n_layers_images/3x8_relu.png')
    
    # Experiment 3 - layer size
    train_moon(
        X, y, 
        [2, 3, 3, 2], 
        filename='n_layers_images/2x3_relu.png')
    train_moon(
        X, y, 
        [2, 100, 100, 2], 
        filename='n_layers_images/2x100_relu.png')
    
    # Experiment 4 - regularization
    train_moon(
        X, y, 
        [2, 100, 100, 2], 
        reg_lambda=0.0, 
        filename='n_layers_images/2x100_relu_no_reg.png')
    train_moon(
        X, y, 
        [2, 100, 100, 2], 
        reg_lambda=0.5, 
        filename='n_layers_images/2x100_relu_reg_0.5.png')

    # ##########################################################
    # Breast cancer dataset
    # ##########################################################
    
    n_features, X_train, X_test, y_train, y_test = generate_cancer_data()
    
    # baseline
    train_cancer(
        X_train, y_train, X_test, y_test,
        [n_features, 20, 10, 2],
        actFun_type='relu',
        name='2x100_relu'
    )
    
    # Experiment 1 - act func
    train_cancer(
        X_train, y_train, X_test, y_test,
        [n_features, 20, 10, 2],
        actFun_type='tanh',
        name='2x100_tanh'
    )
    train_cancer(
        X_train, y_train, X_test, y_test,
        [n_features, 20, 10, 2],
        actFun_type='sigmoid',
        name='2x100_sigmoid'
    )
    
    # Experiment 2 - deep / shallow
    train_cancer(
        X_train, y_train, X_test, y_test,
        [n_features, 200, 2],
        name='1x200_tanh'
    )
    train_cancer(
        X_train, y_train, X_test, y_test,
        [n_features, 50, 50, 50, 2],
        name='3x50_tanh'
    )
    
    # Experiment 3 - layer size
    train_cancer(
        X_train, y_train, X_test, y_test,
        [n_features, 10, 10, 2],
        name='2x10_tanh'
    )
    train_cancer(
        X_train, y_train, X_test, y_test,
        [n_features, 200, 200, 2],
        name='2x200_tanh'
    )
    
    # Experiment 4 - regularization
    train_cancer(
        X_train, y_train, X_test, y_test,
        [n_features, 100, 50, 2],
        reg_lambda=0.0,
        name='2x100_tanh_no_reg'
    )
    train_cancer(
        X_train, y_train, X_test, y_test,
        [n_features, 100, 50, 2],
        reg_lambda=0.5,
        name='2x100_tanh_reg_0.5'
    )

if __name__ == "__main__":
    main()