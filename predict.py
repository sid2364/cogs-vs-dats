print("Please wait...")
import conv_nn
import sys

model = conv_nn.get_model()

while True:
	conv_nn.make_one_prediction(model, 'data/test/%s.jpg' % (input("Enter image number: ")))