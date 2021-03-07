import math
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
            str1 = file.readline()
            arr1 = str1.split(" ")
            self.neuronCount = int(arr1[0])
            self.prevNeuronCount = int(arr1[1])
            for i in range(self.neuronCount):
                str2 = file.readline()
                str2 = str2.replace('[','')
                str2 = str2.replace(']','')
                weightArr = np.fromstring(str2, dtype=float, sep=" ")
                str2 = file.readline()
                biasVal = int(str2)
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
        for neuron in self.neurons:
            neuron.printNeuron()
            
    def saveFile(self):
        file = open(self.layerName+".txt", "a")
        file.write(str(self.neuronCount)+" "+str(self.prevNeuronCount))
        file.write("\n")
        for i in range(self.neuronCount):
            file.write(np.array_str(self.neurons[i].getWeight()))
            file.write("\n")
            file.write(str(self.neurons[i].getBias()))
            file.write("\n")