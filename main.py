import numpy as np
from dataset_input import DatasetInput

data = DatasetInput()
arr = data.readImage(60000)
print(arr)
data.showImage(arr)