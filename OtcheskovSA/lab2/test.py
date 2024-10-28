import numpy as np


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)


def cross_entropy_loss(y_true, y_pred):
    n = y_true.shape[0]
    log_likelihood = -np.log(y_pred[range(n), y_true])
    return np.sum(log_likelihood) / n


def cross_entropy_derivative(y_true, y_pred):
    grad = y_pred.copy()
    grad[range(y_true.shape[0]), y_true] -= 1
    return grad / y_true.shape[0]


def dropout(x, rate, train=True):
    if train:
        mask = (np.random.rand(*x.shape) > rate).astype(float)
        return x * mask / (1.0 - rate)
    return x


class MnistNetMiniBatch:
    def __init__(self):
        self.weights = {
            'W1': np.random.randn(784, 100) * np.sqrt(2. / 784),
            'b1': np.zeros((1, 100)),
            'W2': np.random.randn(100, 50) * np.sqrt(2. / 100),
            'b2': np.zeros((1, 50)),
            'W3': np.random.randn(50, 10) * np.sqrt(2. / 50),
            'b3': np.zeros((1, 10)),
        }

    def forward(self, x, train=True):
        self.cache = {}
        self.cache['A0'] = x

        # Layer 1
        z1 = np.dot(x, self.weights['W1']) + self.weights['b1']
        a1 = relu(z1)
        a1 = dropout(a1, 0.5, train)
        self.cache.update({'Z1': z1, 'A1': a1})

        # Layer 2
        z2 = np.dot(a1, self.weights['W2']) + self.weights['b2']
        a2 = relu(z2)
        a2 = dropout(a2, 0.25, train)
        self.cache.update({'Z2': z2, 'A2': a2})

        # Layer 3 (Output Layer)
        z3 = np.dot(a2, self.weights['W3']) + self.weights['b3']
        a3 = softmax(z3)
        self.cache.update({'Z3': z3, 'A3': a3})

        return a3

    def backward(self, y_true, learning_rate=0.01, mini_batch=True):
        # Gradients for output layer
        m = y_true.shape[0]
        a3 = self.cache['A3']
        dz3 = cross_entropy_derivative(y_true, a3)
        dw3 = np.dot(self.cache['A2'].T, dz3) / m
        db3 = np.sum(dz3, axis=0, keepdims=True) / m

        # Gradients for Layer 2
        da2 = np.dot(dz3, self.weights['W3'].T)
        dz2 = da2 * relu_derivative(self.cache['Z2'])
        dw2 = np.dot(self.cache['A1'].T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        # Gradients for Layer 1
        da1 = np.dot(dz2, self.weights['W2'].T)
        dz1 = da1 * relu_derivative(self.cache['Z1'])
        dw1 = np.dot(self.cache['A0'].T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # Update weights
        self.weights['W3'] -= learning_rate * dw3
        self.weights['b3'] -= learning_rate * db3
        self.weights['W2'] -= learning_rate * dw2
        self.weights['b2'] -= learning_rate * db2
        self.weights['W1'] -= learning_rate * dw1
        self.weights['b1'] -= learning_rate * db1

    def compute_acc(self, X_test, Y_test):
        predictions = np.argmax(self.forward(X_test, train=False), axis=1)
        return np.mean(predictions == Y_test)


# Training loop
def train(net, X_train, Y_train, X_test, Y_test, epochs=100, batch_size=10, learning_rate=0.001):
    for epoch in range(epochs):
        permutation = np.random.permutation(X_train.shape[0])
        X_train_shuffled = X_train[permutation]
        Y_train_shuffled = Y_train[permutation]

        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train_shuffled[i:i + batch_size]
            Y_batch = Y_train_shuffled[i:i + batch_size]

            # Forward pass
            y_pred = net.forward(X_batch)

            # Compute loss
            loss = cross_entropy_loss(Y_batch, y_pred)

            # Backward pass and update weights
            net.backward(Y_batch, learning_rate)

        # Decay learning rate and compute metrics
        learning_rate *= 0.99
        train_acc = net.compute_acc(X_train, Y_train)
        test_acc = net.compute_acc(X_test, Y_test)

        print(f"Epoch {epoch + 1}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")


# Example usage:
# X_train, Y_train, X_test, Y_test = load_data()  # Replace this with your data loading logic
# net = MnistNetMiniBatch()
# train(net, X_train, Y_train, X_test, Y_test)
