import random
import math

class Neuron:
    """
        A simple artificial neuron with ReLU activation.
    """
    def __init__(self, num_inputs):
        # Initialize weights and bias randomly
        self.weights = [random.uniform(-1, 1) for _ in range(num_inputs)]
        self.bias = random.uniform(-1, 1)

    def activate(self, z):
        """ReLU activation function"""
        return max(0, z)

    def forward(self, inputs):
        """
        Compute neuron output:
        z = w·x + b
        """
        z = 0
        for w, x in zip(self.weights, inputs):
            z += w * x
        z += self.bias

        return self.activate(z)
