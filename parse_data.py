'''
Data from: https://www.kaggle.com/c/dogs-vs-cats/data
'''
import cv2
import numpy as np
import os
import pickle

train_d = os.path.join(os.getcwd(), 'data/train/')
test_d = os.path.join(os.getcwd(), 'data/test/')

image_dim = 45 # resizing image to this size (60x60)

def make_dog_cat(dp):
	if dp == 'cat':
		return [0, 1]
	return [1, 0]

def get_dog_cat(arr):
	if arr[0] == 0:
		return 1 #'dog'
	return 0 #'cat'

def read_data_for_one_image(filen):
	file_n = os.path.join(os.getcwd(), filen)
	if os.path.exists(file_n):
		image = cv2.imread(os.path.join(file_n), cv2.IMREAD_GRAYSCALE)
		image = cv2.resize(image, (image_dim, image_dim))
		return image
	print("Image file does not exist.")
	return None

def read_data_and_save():
	print("Reading training data files.")
	training_data = []
	for image_file in os.listdir(train_d):
		cat_or_dog = image_file.split('.')[-3] # cat.1.jpg
		image_id = image_file.split('.')[-2] # not req for training!
		cod_class = make_dog_cat(cat_or_dog)
		image_ = cv2.imread(os.path.join(train_d, image_file), cv2.IMREAD_GRAYSCALE)
		image = cv2.resize(image_, (image_dim, image_dim))
		training_data.append([np.array(image), cod_class]) # [image, class label]

	print("Reading testing data files.")
	testing_data = []
	for image_file in os.listdir(test_d):
		image_id = image_file.split('.')[-2] # 1.jpg
		image_ = cv2.imread(os.path.join(test_d, image_file), cv2.IMREAD_GRAYSCALE)
		image = cv2.resize(image_, (image_dim, image_dim))
		testing_data.append([np.array(image), image_id]) # [image, id]

	#print(training_data[0])
	#print(testing_data[0])

	data = {
		"training_data": training_data,
		"testing_data": testing_data,
	}
	print("Saving processed image data to pkl.")
	with open('data.pkl', 'w') as filep:
		pickle.dump(data, filep)

def get_data():
	if not os.path.exists('data.pkl'):
		read_data_and_save()

	print("Reading processed image data from pkl.")
	with open('data.pkl', 'r') as filep:
		data_r = pickle.load(filep)
		training_data = data_r["training_data"]
		testing_data = data_r["testing_data"]
		#print(training_data[0])
		#print(testing_data[0])
		return training_data, testing_data

