import parse_data
import pickle
import os
import numpy as np
import tflearn
import csv
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# not required coz not jupyter
import tensorflow as tf
tf.reset_default_graph()

image_dim = parse_data.image_dim
training_data, testing_data = parse_data.get_data()
model_path = '11.cogs-and-dats.model'

# Make training set and testing set from the entire labeled training data
test_set = int(len(training_data)*0.8)
X = np.array([t[0] for t in training_data[:test_set]])
Y = np.array([t[1] for t in training_data[:test_set]])
X = X.reshape([-1, image_dim, image_dim, 1])
test_x = np.array([t[0] for t in training_data[test_set:]])
test_y = np.array([t[1] for t in training_data[test_set:]])
test_x = test_x.reshape([-1, image_dim, image_dim, 1])

# Make array from testing data that can be passed for prediction
predict_this_X = np.array([t[0] for t in testing_data])
predict_this_X = predict_this_X.reshape([-1, image_dim, image_dim, 1])
predict_this_X_id = np.array([t[1] for t in testing_data])

#print(image_dim)
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
	
	convnet = fully_connected(convnet, 1024, activation='relu')
	convnet = dropout(convnet, 0.8)

	convnet = fully_connected(convnet, 2, activation='softmax')
	convnet = regression(convnet, optimizer='adam', metric='accuracy', learning_rate=0.001, loss='categorical_crossentropy', name='targets')

	model = tflearn.DNN(convnet)

	return model

def fit_model(model, X, Y, test_x, test_y):
	print("Fitting model with data.")
	model.fit({'input': X}, {'targets': Y}, n_epoch=12, validation_set=({'input': test_x}, {'targets': test_y}), 
		snapshot_step=500, show_metric=True, run_id='cats-vs-dogs')
	model.save(model_path)

	return model

def get_model():
	print("Getting the model.")
	model = make_model()
	if not os.path.exists(model_path + ".index"):
		print(os.path.join('models', model_path) + " does not exist!")
		fit_model(model, X, Y, test_x, test_y)
	else:
		print(model_path + " exists! Loading the model.")
		model.load(model_path)
	return model

def make_predictions(model, test_arr):
	print("Making predictions!")
	return model.predict(test_arr)	

def get_higher_proba(prediction):
	# [1, 0] is dog, [0, 1] is cat
	#prediction.reshape(-1, 2)
	#print(prediction)
	if prediction[0] > prediction[1]:
		return 1
	return 0

def make_one_prediction(model, filen):
	print("Reading image data.")
	test_arr = parse_data.read_data_for_one_image(filen)
	test_arr = test_arr.reshape([-1, image_dim, image_dim, 1])
	if test_arr is None:
		print("That's not a valid file.")
		return
	prediction = model.predict(test_arr)[0]

	if get_higher_proba(prediction) == 0:
		print("==== That is a cat! ====")
	else:
		print("==== That's a dog! ====")
	print("Dog %s vs. Cat %s" % (prediction[0], prediction[1]))

if __name__ == "__main__":
	model = get_model()
	predicted_data = make_predictions(model, predict_this_X)
	final = []
	with open('submission.csv', 'w') as filep:
		writer = csv.writer(filep, delimiter=',')
		writer.writerow(['id', 'label'])
		for i in range(len(predicted_data)):
			final.append([predict_this_X_id[i], predicted_data[i]])
			print([predict_this_X_id[i], predicted_data[i], get_higher_proba(predicted_data[i])])
			writer.writerow([predict_this_X_id[i], get_higher_proba(predicted_data[i])])

	with open('predictions.pkl', 'w') as filep:
		pickle.dump(final, filep)
