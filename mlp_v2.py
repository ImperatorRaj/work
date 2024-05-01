import numpy as np

class NeuralNetwork:
	def __init__(self, input_size, hidden_size, output_size):
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size

		# Initialize weights
		self.weights_input_hidden = np.array([[0.15, 0.2], [0.25, 0.3]])
		self.weights_hidden_output = np.array([[0.4, 0.45], [0.5, 0.55]])

		# Initialize the biases
		self.bias_hidden = np.array([0.35, 0.35])
		self.bias_output = np.array([0.6,0.6])

	def sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	def sigmoid_derivative(self, x):
		return x * (1 - x)

	def feedforward(self, X):
		# Input to hidden
		self.hidden_activation = np.dot(X, self.weights_input_hidden) + self.bias_hidden
		self.hidden_output = self.sigmoid(self.hidden_activation)

		# Hidden to output
		self.output_activation = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
		self.predicted_output = self.sigmoid(self.output_activation)

		return self.predicted_output

	def backward(self, X, y, learning_rate):
		# Compute the output layer error
		output_error = y - self.predicted_output
		output_delta = output_error * self.sigmoid_derivative(self.predicted_output)

		# Compute the hidden layer error
		hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
		hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)

		# Update weights and biases
		self.weights_hidden_output += np.dot(self.hidden_output.T, output_delta) * learning_rate
		self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
		self.weights_input_hidden += np.dot(X.T, hidden_delta) * learning_rate
		self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

	def train(self, X, y, epochs, learning_rate):
		for epoch in range(epochs):
			output = self.feedforward(X)
			self.backward(X, y, learning_rate)
			if epoch % 1 == 0:
				loss = np.mean(np.square(y - output))
				print(f"Epoch {epoch}, Loss:{loss}, Input to hidden layer wts: {self.weights_input_hidden}, Hidden to output layer wts: {self.weights_hidden_output}")

X = np.array([0.05, 0.10])
y = np.array([0.01,0.99])

nn = NeuralNetwork(input_size=2, hidden_size=2, output_size=2)
nn.train(X, y, epochs=1, learning_rate=0.5)


output = nn.feedforward(X)
print("Predictions after training:")
print(output)
