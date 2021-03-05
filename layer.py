import math
import numpy as np
from neuron import Neuron
#This class is for a single layer and holds necessary functions for a single layer 
class Layer:
    #Initialize layer with random weights and bias
    def __init__(self, neuronCount, prevNeuronCount=0, input=[], weight=[], bias=[], activation="relu"):
        self.neuronCount = neuronCount
        self.prevNeuronCount = prevNeuronCount
        self.input = input
        self.weight = weight
        self.bias = bias
        self.activation = activation
        self.neurons = []
        self.output = []
        if self.prevNeuronCount > 0:
            for i in range(self.neuronCount):
                neuron = Neuron(input=np.random.rand(self.prevNeuronCount), 
                                weight=np.random.rand(self.prevNeuronCount),
                                bias=np.random.randint(-5,5))
                self.neurons.append(neuron)
        else:
            for i in range(self.neuronCount):
                neuron = Neuron(input=self.input[i], weight=self.weight[i], bias=self.bias[i])
                self.neurons.append(neuron)

    def calculateOutput(self):
        for i in range(self.neuronCount):
            self.neurons[i].calculateOutput()
    #Relu is done on neuron level but softmax is done on layer level
    def activateLayer(self):
        self.calculateOutput()
        if self.activation == "relu":
            for i in range(self.neuronCount):
                self.neurons[i].relu()
        elif self.activation == "softmax":
            maxOutput = self.neurons[0].getOutput()
            sumOutput = 0
            for i in range(self.neuronCount):
                if maxOutput < self.neurons[i].getOutput():
                    maxOutput = self.neurons[i].getOutput()
            for i in range(self.neuronCount):
                self.neurons[i].setOutput(self.neurons[i].getOutput()-maxOutput)
                self.neurons[i].setOutput(math.e**self.neurons[i].getOutput())
                sumOutput+=self.neurons[i].getOutput()
            for i in range(self.neuronCount):
                self.neurons[i].setOutput(self.neurons[i].getOutput()/sumOutput)

    def getOutput(self):
        output = []
        for i in range(self.neuronCount):
            output.append(self.neurons[i].getOutput())
        self.output = output
        return self.output