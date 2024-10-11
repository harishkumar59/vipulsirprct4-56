import numpy as np

class NeuralNerwork():
    
    def __init__(self):
        np.random.seed(1)
        
        self.synaptic_weights = 2 * np.random.random((3,1)) -1

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self,x):
        
        return x * (1 - x)
    #from here tabhnine code for
    def train(self, training_inputs, training_outputs, num_iterations):
        for iteration in range(num_iterations):
            outputs = self.think(training_inputs)
            
            error = training_outputs - outputs
            
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(outputs))
            
            self.synaptic_weights += adjustments

    def think(self, inputs):
        # return self.sigmoid(np.dot(inputs, self.synaptic_weights))

        inputs = inputs.astype(float)

        output = self.sigmoid(np.dot(inputs,self.synaptic_weights))

        return output
    
if __name__ == "__main__":
    neural_network = NeuralNerwork()
    
    print("Random starting synaptic weights:")
    print(neural_network.synaptic_weights)
    
    training_inputs = np.array([[0,0,1], [1,1,1], [1,0,1], [0,1,1]])
    training_outputs = np.array([[0,1,1,0]]).T
    
    neural_network.train(training_inputs, training_outputs, 10000)
    
    print("New synaptic weights after training:")
    print(neural_network.synaptic_weights)


    A = str(input("input 1: "))
    B = str(input("input 2: "))
    C = str(input("input 3: "))

    # inputs = np.array([[float(A), float(B), float(C)]])
    # print("Predicted output:")
    # print(neural_network.think(inputs))

    print("New situatijon: input data =", A,B,C)
    print("Predicted output:")
    print(neural_network.think(np.array([A,B,C])))

    
