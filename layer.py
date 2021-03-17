import math
import json
import numpy as np
from neuron import Neuron

#This class is for a single layer and holds necessary functions for a single layer 
class Layer:

    #Initialize layer with random weights and bias
    def __init__(self, layerName, inputArr, neuronCount = 0, prevNeuronCount = 0, activation="relu", mode="init_rand"):
        self.layerName = layerName
        self.inputArr = inputArr
        self.activation = activation
        self.mode = mode
        self.neurons = []
        if mode == "init_rand":
            self.neuronCount = neuronCount
            self.prevNeuronCount = prevNeuronCount
            for i in range(self.neuronCount):
                neuron = Neuron(input=self.inputArr, 
                                weight=np.random.rand(self.prevNeuronCount),
                                bias=np.random.randint(-5,5))
                self.neurons.append(neuron)
        elif mode == "init_read":
            file = open(self.layerName+".txt","r")
            data = json.load(file)
            self.neuronCount = data["neuronCount"]
            self.prevNeuronCount = data["prevNeuronCount"]
            for i in range(self.neuronCount):
                weightArr = data["weight"+str(i+1)]
                biasVal = data["bias"+str(i+1)]
                neuron = Neuron(self.inputArr, weightArr, biasVal)
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

    def printLayer(self):
        print("neuronCount: ", self.neuronCount)
        print("prevNeuronCount", self.prevNeuronCount)
        for neuron in self.neurons:
            neuron.printNeuron()
            
    def saveLayer(self):
        data = {}
        data["neuronCount"] = self.neuronCount
        data["prevNeuronCount"] = self.prevNeuronCount
        for i in range(self.neuronCount):
            weightArr = self.neurons[i].getWeight()
            weightArr = list(weightArr)
            biasVal = self.neurons[i].getBias()
            data["weight"+str(i+1)] = weightArr
            data["bias"+str(i+1)] = biasVal
        print(data)
        file = open(self.layerName+".txt", "a")
        json.dump(data, file)
        file.close()