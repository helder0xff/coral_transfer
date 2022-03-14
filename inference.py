import tensorflow as tf
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

interpreter.invoke()

start_global = datetime.datetime.now()
interpreter.invoke()
end_global = datetime.datetime.now()
delta = end_global - start_global
print(delta.total_seconds()*1000/(10), 'mS')

output_details = interpreter.get_output_details()[0]
output = interpreter.get_tensor(output_details['index'])
scale, zero_point = output_details['quantization']
output = scale * (output - zero_point)
output = np.argmax(output)

if(0 == output):
	print('Rose')
else:
	print('Sunflower')

#### end of file ####
