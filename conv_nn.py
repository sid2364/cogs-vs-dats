import parse_data
import os
import numpy as np
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

import tensorflow as tf
#tf.reset_default_graph()

image_dim = parse_data.image_dim
training_data, testing_data = parse_data.get_data()
model_path = '5.cogs-and-dats.model'

#print(training_data[0])
#print(testing_data[0])

test_set = int(len(training_data)*0.80)

X = np.array([t[0] for t in training_data[:test_set]])
Y = []
for t in training_data[:test_set]:
	if t[1] == 0:
		Y.append([1,0])
	else:
		Y.append([0,1])
#Y = np.array(Y)

test_x = np.array([t[0] for t in training_data[test_set:]])
test_y = []
for t in training_data[test_set:]:
	if t[1] == 0:
		test_y.append([1, 0])
	else:
		test_y.append([0, 1])
#test_y = np.array(test_y)

X = X.reshape([-1, image_dim, image_dim, 1])
test_x = test_x.reshape([-1, image_dim, image_dim, 1])

#Y = Y.reshape([-1, 2])
#test_y = test_y.reshape([-1, 2])

print(X.shape)
#print(Y.shape)

print(image_dim)
def make_model():
	print("Making the model.")
	convnet = input_data(shape=[None, image_dim, image_dim, 1], name='input')

	convnet = conv_2d(convnet, 32, 5, activation='relu')
	convnet = max_pool_2d(convnet, 5)

	convnet = conv_2d(convnet, 128, 5, activation='relu')
	convnet = max_pool_2d(convnet, 5)

	convnet = conv_2d(convnet, 256, 5, activation='relu')
	convnet = max_pool_2d(convnet, 5)

	convnet = conv_2d(convnet, 128, 5, activation='relu')
	convnet = max_pool_2d(convnet, 5)

	convnet = conv_2d(convnet, 64, 5, activation='relu')
	convnet = max_pool_2d(convnet, 5)
	
	convnet = fully_connected(convnet, 1024, activation='relu')
	convnet = dropout(convnet, 0.8)

	convnet = fully_connected(convnet, 2, activation='softmax')
	convnet = regression(convnet, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy', name='targets')

	model = tflearn.DNN(convnet)

	return model

def fit_model(model, X, Y, test_x, test_y):
	print("Fitting model with data.")
	model.fit({'input': X}, {'targets': Y}, n_epoch=10, validation_set=({'input': test_x}, {'targets': test_y}), 
		snapshot_step=500, show_metric=True, run_id='cats-vs-dogs')
	model.save(os.path.join("./models", model_path))

	return model

def get_model():
	print("Getting the model.")
	if not os.path.exists(os.path.join('models', model_path + ".index")):
		print(os.path.join('models', model_path) + " does not exist!")
		model = make_model()
		fit_model(model, X, Y, test_x, test_y)
	else:
		print(os.path.join('models', model_path) + " exists! Loading the model.")
		model.load(os.path.join('model', model_path))
	return model

if __name__ == "__main__":
	m = get_model()