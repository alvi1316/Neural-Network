import gzip
import numpy as np
import matplotlib.pyplot as plt

class DatasetInput:

    def __init__(self):
        self.file = gzip.open('train-images-idx3-ubyte.gz','r')

    def readImage(self, imageIndex):
        if imageIndex > 59999 or imageIndex < 0:
            print("Image index is out of bound.")
        else:
            self.file.seek(16+(imageIndex*784))
            buffer = self.file.read(784)
            data = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32)
            return data

    def showImage(self, pixelArray):
        if pixelArray != None:   
            pixelArray = pixelArray.reshape(28, 28, 1)
            image = np.asarray(pixelArray).squeeze()
            plt.imshow(image)
            plt.show()
        else:
            print("Pixel array is None.")

    def close(self):
        self.file.close()