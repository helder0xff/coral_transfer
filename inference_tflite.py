import tflite_runtime.interpreter as tflite
from PIL import Image
import numpy as np
import datetime

IMAGE_SIZE 		= 224
IMAGE_CHANNELS 	= 3
IMAGE_SHAPE 	= (IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS)

input_image = Image.open('rose.jpg')
input_array = np.asarray(input_image)
input_array = np.resize(input_array, IMAGE_SHAPE)
input_array = np.expand_dims(input_array, axis=0)

interpreter = tf.lite.Interpreter('mobilenet_v2_1.0_224_quant.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
interpreter.set_tensor(input_details['index'], input_array)

start_global = datetime.datetime.now()
for i in range(10):
	interpreter.invoke()
end_global = datetime.datetime.now()
delta = end_global - start_global
print(delta.total_seconds()*1000/(10), 'mS')

