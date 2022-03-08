import tensorflow as tf
import os
import numpy as np

################################################################################
#
# CONSTANTS
#
################################################################################
IMAGE_SIZE 		= 224
IMAGE_CHANNELS 	= 3
IMG_SHAPE 		= (IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS)
BATCH_SIZE 		= 64

################################################################################
#
# CONFIG.
#
################################################################################
datasets_path 	= '/media/helder/workspace/datasets'
dataset_path    = datasets_path + '/flower_photos'                       

################################################################################
#
# CREATE BATCH FLOW
#
################################################################################
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
	rescale 			= 1./255, 
    validation_split 	= 0.2)	# Set validation set to 20%

train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size = (IMAGE_SIZE, IMAGE_SIZE),
    batch_size 	= BATCH_SIZE, 
    subset 		= 'training')

val_generator = datagen.flow_from_directory(
    dataset_path,
    target_size = (IMAGE_SIZE, IMAGE_SIZE),
    batch_size 	= BATCH_SIZE, 
    subset 		= 'validation')

################################################################################
# To iterate through the generator just use the build in next method this way:
#
#	image_batch, label_batch = next(val_generator)
#
# Each time we invoke 'next' it will return a batch of images and the label for
# each image.
################################################################################

################################################################################
#
# THE MODEL
#
################################################################################
# Get the base model (MobileNet V2) ############################################
base_model = tf.keras.applications.MobileNetV2(	input_shape = IMG_SHAPE,
                                              	include_top = False, 
                                              	weights 	='imagenet')
base_model.trainable = False

# Complete the base model for our own purpose ##################################
number_classes = len(train_generator.class_indices)

model = tf.keras.Sequential([
  base_model,
  tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation = 'relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(units = number_classes, activation = 'softmax')
])

# Configure the model ##########################################################
model.compile(	optimizer 	='adam', 
              	loss		='categorical_crossentropy', 
              	metrics		=['accuracy'])

print(model.summary())

# Train the model ##############################################################
if False == os.path.isfile('mobilenet_v2_1.0_224_quant.tflite'):
	history = model.fit(train_generator,
	                    steps_per_epoch 	= len(train_generator), 
	                    epochs				= 10,
	                    validation_data		= val_generator,
	                    validation_steps	= len(val_generator))

	############################################################################
	# Quantize the model
	############################################################################
	# A generator that provides a representative dataset
	def representative_data_gen():
	  dataset_list = tf.data.Dataset.list_files(dataset_path + '/*/*')

	  for i in range(100):
	    image = next(iter(dataset_list))
	    image = tf.io.read_file(image)
	    image = tf.io.decode_jpeg(image, channels=3)
	    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
	    image = tf.cast(image / 255., tf.float32)
	    image = tf.expand_dims(image, 0)
	    yield [image]

	converter = tf.lite.TFLiteConverter.from_keras_model(model)
	# This enables quantization
	converter.optimizations = [tf.lite.Optimize.DEFAULT]
	# This sets the representative dataset for quantization
	converter.representative_dataset = representative_data_gen
	# This ensures that if any ops can't be quantized, 
	# the converter throws an error
	converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
	# For full integer quantization, though supported types defaults to int8 
	# only, we explicitly declare it for clarity.
	converter.target_spec.supported_types = [tf.int8]
	# These set the input and output tensors to uint8 (added in r2.3)
	converter.inference_input_type = tf.uint8
	converter.inference_output_type = tf.uint8

	# Convert the model to TFLite ##############################################
	tflite_model = converter.convert()

	with open('mobilenet_v2_1.0_224_quant.tflite', 'wb') as f:
	  f.write(tflite_model)

################################################################################
#
# CHECK PERFORMANCE
#
################################################################################
def set_input_tensor(interpreter, input):
  input_details = interpreter.get_input_details()[0]
  tensor_index = input_details['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  # Inputs for the TFLite model must be uint8, so we quantize our input data.
  # NOTE: This step is necessary only because we're receiving input data from
  # ImageDataGenerator, which rescaled all image data to float [0,1]. When using
  # bitmap inputs, they're already uint8 [0,255] so this can be replaced with:
  #   input_tensor[:, :] = input
  scale, zero_point = input_details['quantization']
  input_tensor[:, :] = np.uint8(input / scale + zero_point)

def classify_image(interpreter, input):
  set_input_tensor(interpreter, input)
  interpreter.invoke()
  output_details = interpreter.get_output_details()[0]
  output = interpreter.get_tensor(output_details['index'])
  # Outputs from the TFLite model are uint8, so we dequantize the results:
  scale, zero_point = output_details['quantization']
  output = scale * (output - zero_point)
  top_1 = np.argmax(output)

  return top_1

batch_images, batch_labels = next(val_generator)
interpreter = tf.lite.Interpreter('mobilenet_v2_1.0_224_quant.tflite')
interpreter.allocate_tensors()

# Collect all inference predictions in a list
batch_prediction = []
batch_truth = np.argmax(batch_labels, axis=1)

for i in range(len(batch_images)):
  prediction = classify_image(interpreter, batch_images[i])
  batch_prediction.append(prediction)

# Compare all predictions to the ground truth
tflite_accuracy = tf.keras.metrics.Accuracy()
tflite_accuracy(batch_prediction, batch_truth)
print("Quant TF Lite accuracy: {:.3%}".format(tflite_accuracy.result()))
#### end of file ####
