import numpy as np

class Perceptron:

    def __init__(self):
        self.weights = None
        self.bias = 0

    def initialize(self, n_features,w1,w2,bias):

        self.weights = np.zeros(n_features)
        self.weights[0] = w1
        self.weights[1]=w2
        self.bias = bias
        return

    def predict(self, inputs):

        activation = np.dot(inputs, self.weights) + self.bias
        return 1 if activation > 0 else 0

    def train(self, X, y, epochs ,learning_rate):
        # initialize the w and b
        #self.initialize(X.shape[1])
        for epoch in range(epochs):
            for inputs, label in zip(X, y):
                # get prediction
                y_pred = self.predict(inputs)
                # calculate delta error
                error = label - y_pred
                # update w and b
                self.weights += learning_rate * error * inputs
                self.bias += learning_rate * error
            if (epoch+1)%10==0:
              print("Epoch : ",epoch+1," Weights: ",self.weights," Bias: ",self.bias)
        return


# AND

X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([0, 0, 0, 1])

init_w1 = float(input("Enter the initial weight_1: "))
init_w2 = float(input("Enter the initial weight_2: "))
bias = float(input("Enter the initial bias: "))

p = Perceptron()
p.initialize( 2, init_w1, init_w2, bias)
p.train(X_train, y_train,epochs=100, learning_rate=0.1)
test_input = np.array([1, 0])
print("Resultant output: ",p.predict(test_input))  # Output: 0

# OR

X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([0, 1, 1, 1])

p = Perceptron()
print("init1: ",init_w1," init_w2: ",init_w2)
p.initialize(2, init_w1, init_w2, bias)
p.train(X_train, y_train, epochs=100, learning_rate=0.1)
test_input = np.array([0, 1])
print("Resultant output: ",p.predict(test_input))  # Output: 1
