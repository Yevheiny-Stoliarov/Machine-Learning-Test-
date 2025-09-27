import random
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def init_layer(input_size, output_size):
    return [[random.uniform(-1, 1) for _ in range(input_size)] for _ in range(output_size)]

def dot(inputs, weights):
    return [sum(i * w for i, w in zip(inputs, neuron)) for neuron in weights]

def feedforward(inputs, weights1, weights2):
    hidden_raw = dot(inputs, weights1)
    hidden_activated = [sigmoid(x) for x in hidden_raw]
    output_raw = dot(hidden_activated, weights2)
    output_activated = [sigmoid(x) for x in output_raw]
    return hidden_activated, output_activated

def train(data, weights1, weights2, lr=0.1, epochs=1000):
    for _ in range(epochs):
        for inputs, target in data:
            hidden, output = feedforward(inputs, weights1, weights2)

            error = [target[i] - output[i] for i in range(len(target))]
            d_output = [error[i] * sigmoid_derivative(output[i]) for i in range(len(output))]

            error_hidden = []
            for i in range(len(weights2[0])):
                err = sum(d_output[j] * weights2[j][i] for j in range(len(weights2)))
                error_hidden.append(err * sigmoid_derivative(hidden[i]))

            for i in range(len(weights2)):
                for j in range(len(weights2[i])):
                    weights2[i][j] += lr * d_output[i] * hidden[j]

            for i in range(len(weights1)):
                for j in range(len(weights1[i])):
                    weights1[i][j] += lr * error_hidden[i] * inputs[j]

def predict(inputs, weights1, weights2):
    _, output = feedforward(inputs, weights1, weights2)
    return output

training_data = []
for _ in range(100):
    a = random.randint(0, 10)
    b = random.randint(0, 10)
    label = [1] if a + b > 10 else [0]
    training_data.append(([a / 10, b / 10], label))

weights1 = init_layer(2, 4)  # 2 inputs → 4 hidden neurons
weights2 = init_layer(4, 1)  # 4 hidden → 1 output

train(training_data, weights1, weights2)

test_input = [0.6, 0.5]  # 6 + 5 = 11 → should be 1
prediction = predict(test_input, weights1, weights2)
print("Test Input:", test_input)
print("Prediction:", round(prediction[0], 2))
