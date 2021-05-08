import tensorflow.keras as keras
import numpy as np
from tensorflow.keras.layers import Embedding

input = np.random.randint(1000, size=(32, 10))

model = keras.Sequential()
model.add(Embedding(2000, 64))
model.compile('rmsprop', 'mse')

output = model.predict(input)
print(input.shape, output.shape)
