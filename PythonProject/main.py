#!/bin/python
import neuralNetwork
import torch
import numpy as np
from PIL import Image
from GUI import GUI
import globalVars


class Application(object):
	def __init__(self):
		# All set-up code must run __before__ this line
		# No code will run after this line until the GUI is closed by the user
		# self.GUI = GUI(self.model, self.process_func)
		self.GUI = GUI(self.active_model_func, self.process_func)

	# Here we will retrieve the image from the drawing canvas,
	# feed the image into the neural network and gather the output data
	# then update the UI accordingly
	#Based from: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
	def process_func(self, image: np.ndarray):
		image = Image.fromarray(np.uint8(image))
		# Resize whilst maintaining aspect ratio
		# the MNIST images are centered in the 28x28 square, and the
		# actual drawing is always within 20x20 pixels
		# thus each digit in the MNIST set has 4 pixels padding on either side
		# in either the vertical or horizontal direction
		try:
			x, y, w, h = image.getbbox()
		except TypeError:
			globalVars.predictionResults = None
			globalVars.predictedNum = None
			return  # The user has not drawn an image (the canvas is blank)
		# we want 4 px on each side of the drawing after resizing to 28x28
		# so calculate how many pixels of padding we need to add to each sde before
		# the resize to end up with 4 after the resize
		xpadding = image.size[0] // 28 * 4
		ypadding = image.size[1] // 28 * 4
		print(xpadding, ypadding)
		# Add 4 pixels of padding
		x -= xpadding; y -= ypadding; w += xpadding; h += ypadding
		image = image.crop((x, y, w, h))
		# resize the image to 28x28 (what the neural network expects)
		image = image.resize((28, 28))
		#image.show()

		image = neuralNetwork.transforms.ToTensor()((image))
		# If it is a convolution neural network, we need 4 dimensions
		# (for batch, channel, height, width)
		batch = image.unsqueeze(0) if globalVars.model.is_conv else image


		# Turn of training for the network, we only want to evaluate
		globalVars.model.eval()
		with torch.no_grad():
			# Make the prediction
			results = globalVars.model(batch)
		# Find the weightings and the number class they correspond to
		print(f'Recognised Number Classes: {results.argmax()}')
		print(results)
		globalVars.predictedNum = int(results.argmax())
		# Normalise the results to the range 0...1 (regular probability values)
		globalVars.predictionResults = torch.nn.Softmax(dim=1)(results)[0]
		# Turn training back on
		globalVars.model.train()
		pass

	# Callback function for the GUI indicate the user
	# wants to select/create a new model
	# attach_model_func is a callback function we can use to tell the GUI what the active model is
	def active_model_func(self, attach_model_func, type):
		if type.lower() == 'none':
			globalVars.model = None
		else:
			# Create an instance of the specified model type
			globalVars.model = neuralNetwork.__dict__[type]()
		attach_model_func()


# Create the main window
if __name__ == "__main__":
	Application()
