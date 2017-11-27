import conv_nn
import sys

model = conv_nn.get_model()
conv_nn.make_one_prediction(model, 'data/test/%s.jpg' % (sys.argv[1]))