import numpy as np

#This class is for a single neuron and holds necessary functions for a single neuron
class Neuron:
    
    def __init__(self ,input=[], weight=[], bias=0):
        self.input = input
        self.weight = weight
        self.bias = bias
        self.output = None

    def setInput(self, input):
        self.input = input

    def setWeight(self, weight):
        self.weight = weight

    def setBias(self, bias):
        self.bias = bias

    #This function is necessary for softmax function
    def setOutput(self, output):
        self.output = output

    #Calculation is also done on neuron level
    def calculateOutput(self):
        self.output = np.dot(self.input,self.weight) + self.bias

    #Relu is executed on neuron level
    def relu(self):
        if self.output <= 0:
            self.output = 0
    
    def getOutput(self):
        return self.output

    def getWeight(self):
        return self.weight

    def getBias(self):
        return self.bias

    def printNeuron(self):
        print("Input: ", self.input)
        print("Weight: ", self.weight)
        print("Bias: ", self.bias)
        print("Output: ", self.output)
