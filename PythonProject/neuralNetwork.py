from torch.utils import data
from torchvision import transforms, datasets
import torch
import torch.nn as nn
import torch.optim as optim

# Local imports
import globalVars

# Python Standard Library
import os
import time
from urllib import request
from urllib.error import URLError
import zipfile
import shutil
from glob import glob

# Path were the MNIST dataset will be stored
data_path = 'mnist_data/'
cache_path = 'Model_Cache'

# (will be) The image sets 'as is'
image_sets = None
# (will be) the image sets shuffled and batched for input to the neural network
image_dataset = None


# On complete is an optional callback function that is called when
# data loading is complete
def load_data(on_complete=None):
	global image_sets
	global image_dataset

	batch_size = 4
	print("Creating dataset")

	# MNIST Dataset (essentially the images as tensors)
	train_dataset = datasets.MNIST(root=data_path, train=True, transform=transforms.ToTensor())
	test_dataset = datasets.MNIST(root=data_path, train=False, transform=transforms.ToTensor())

	# Data Loader (Input Pipeline, in batches etc.)
	train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
	test_loader = data.DataLoader(dataset=test_dataset,   batch_size=batch_size, shuffle=False)

	# The image sets 'as is'
	image_sets = (train_dataset, test_dataset)
	# the image sets shuffled and batched for input to the neural network
	image_dataset = (train_loader, test_loader)

	print("Data all loaded")

	if on_complete: on_complete()


# We already have the data downloaded, let's load it
if os.path.exists(data_path):
	load_data()


# Remove all downloaded mnist training/testing data
def rm_dload_data():
	try:
		shutil.rmtree(data_path)
		os.remove('mnist_data.zip')
	except FileNotFoundError:
		pass


# Remove model cache data (by default remove all cached data)
def rm_cache_data(classname="*"):
	file_list = glob(f'{cache_path}/{classname}.pt')
	for f in file_list:
		os.remove(f)

# Download mnist training and testing data from Google Drive.
# The file is hosted on Google Drive
#
# callback_func (if supplied) is called with a signal integer argument indicating
# how far the download has progressed (from 0 - 100)
def download_data(callback_func=None):
	# API documented @: https://developers.google.com/drive/api/v3/reference/files/get
	api_url = 'https://www.googleapis.com/drive/v3/files/'
	# API Key

	api_key = 'AIzaSyAZZD_uyYVy1eJ-KlGRccb9ccuQuQlNhJI'
	# ID for the test README.txt in the Google Drive
	test_id = '1KVa8ISE-a8Oxli1SJH4b7oqkRVD-qM9u'
	# ID for the actual MNIST dataset ZIP file
	file_id = '1VFNqqm9XvcDg9carr8Q2JRx8KSIg0sJk'

	test_url = f'{api_url}{test_id}?key={api_key}&alt=media'
	final_url = f'{api_url}{file_id}?key={api_key}&alt=media'

	local_zip = 'mnist_data.zip'

	print('HTTP Requesting Google Drive API')
	# We're just starting off, call the callback with 5%
	if callback_func: callback_func(5)

	try:
		response = request.urlopen(final_url)
	except URLError as e:
		print(f'Error getting MNIST data from GDrive: {e}')
		return

	print('Received Google Drive Response')
	print('attempting to download the mnist data from google drive')
	# We're getting about to download, call the callback with 10%
	if callback_func: callback_func(10)

	# Save the response to disk, and the file and then proceed to unzip it
	# normally we would use shutil.copyfileobj() to do something like this,
	# but that function provides no feedback in terms of progress.
	# So we implementz our own copy routine loosely based on the original
	# standard python function
	buf_length = 64 * 1024
	file_length = int(response.headers['content-length'])
	print('Download file length:', file_length, 'bytes')
	with open(local_zip, 'wb') as f:
		count = 0  # Number of bytes written
		last_count = 0
		while True:
			buf = response.read(buf_length)
			if not buf:
				break  # We've read the entire file
			# copy to local file
			count += f.write(buf)
			# We have 80% progress to use for the download,
			# divide it evenly among the bytes!
			# (multiply by 70 add 10 to keep between 10% and 80%)
			# Only call the callback function if we have written more then
			# 0.1 MB since last time we called it
			# (to give the Qt event loop time to process events)
			if callback_func and count - last_count > 100000:
				callback_func(int((count / file_length) * 70 + 10))
				last_count = count

	print('Download Complete')

	# Now we can unzip the file with Python's built in Zip utilities
	# We're extracting, call the callback with 90%
	if callback_func: callback_func(90)
	with zipfile.ZipFile(local_zip) as zf:
		zf.extractall()
	# Delete the zip file we downloaded
	os.remove(local_zip)

	# We're all done, call the callback with 100%
	if callback_func: callback_func(100)

	# Load the data
	load_data()

# All models should inherit form this class
# This class defines the common code and behaviour that all models should have
# The __init__ function of the model provides a way to adjust all paramaters
# to quickly create and prototype new models. All code that all models should
# share/execute is to be placed in this class.
#
#
# When crating an instance of this class, the model subclass should call the
# __init__ method of this class in it's __init__ (super().__init__())
# and pass in the following:
#	- layer_list: a list of INSTANCES of pytorch layers (i.e.: nn.Linear())
#	- activation_class: the NAME of the activation class to use (i.e.: nn.ReLU)
#	- epochs: the number of ephocs to trian this model with (i.e.: 10)
#	- loss_class: the NAME of the loss function class (i.e.: nn.CrossEntropyLoss)
#	- optimiser_class: the NAME of the optimiser class (i.e.: optim.Adam)
#	- learning_rate: the learning rate paramater for the optimiser class (i.e.: 0.001)
#	- modelInfo is a string that has a (user friendly) description of the model
#
class NeuralModel(nn.Module):
	# Note pass in the name of all classes, NOT an instance of the classes
	# EXCEPT the layer_list, which should be a list of instances of the lauyer classes
	def __init__(self, layer_list, activation_class, epochs, loss_class, optimiser_class, learning_rate, modelInfo=None):
		self.trained = False
		# make these properties of the model (used by the GUI) an alias
		# to the function/global variables above
		self.download_data = download_data
		self.rm_dload_data = rm_dload_data
		self.image_sets = image_sets

		self.accuracy = None
		self.trainTime = None
		self.modelInfo = modelInfo

		super().__init__()
		print("Training Model being prepared")

		# define the layers
		for idx, func in enumerate(layer_list):
			setattr(self, f'layer{idx}', func)

		# Is it a convolutional network ?
		self.is_conv = any(1 if isinstance(i, nn.Conv2d) else 0 for i in layer_list)
		self.layer_list = layer_list
		self.activation_func = activation_class()
		self.epochs = epochs
		self.loss_func = loss_class()
		self.learning_rate = learning_rate
		self.optimiser_func = optimiser_class(self.parameters(), learning_rate)

	def forward(self, img):
		# We must transform the img (x) differently depending on if this is
		# a convolutional or linear network
		def transform(input):
			if self.is_conv:
				# if ndim is 2, we have already done this
				# if ndim is 3, we are being passed an image
				# from the recognise function and we need to
				# add an extra dimension for the batch size (which is 1 image)
				# if ndim is 4 then we are dealing with the training images,
				if input.ndim == 4:
					# Make a 2D tensor of [Batch images, flattened image]
					return input.view(input.size()[0], -1)
				elif input.ndim == 3:
					return input.unsqueeze(0)

			else:
				# if ndim is 2, we have already done this
				if input.ndim > 2:
					# Flatten the image (and batches if any)
					return input.view(-1, 28 * 28)
			# If we got here, we've done nothing - just return the original
			return input

		x = img
		for idx, func in enumerate(self.layer_list[:-1]):
			layer = getattr(self, f'layer{idx}')
			if isinstance(layer, nn.Linear):
				x = self.activation_func(layer(transform(x)))
			elif isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Conv2d):
				x = self.activation_func(layer(x))
		x = transform(x)
		return getattr(self, f'layer{len(self.layer_list)-1}')(x)

	def test(self, progress_func=None, report_time=False):
		# We can't test without the mnist data being loaded
		if not self.is_data_loaded():
			return

		start_time = time.time()
		def call_progress_func(percent):
			if report_time: progress_func(percent, round(time.time() - start_time, 2))
			elif progress_func: progress_func(percent)
			else: return

		print("Finding accuracy")
		test_dataset = image_dataset[1]
		count = 0
		correct = 0
		total = 0
		total_images = 10000
		# Used to help fund accuracy
		with torch.no_grad():
			for data in test_dataset:
				X, y = data
				output = self(X)  # Passing in flattened image

				for idx, i in enumerate(output):
					if torch.argmax(i) == y[idx]:
						correct += 1
					total += 1
				call_progress_func(100 * count / total_images)
				count = count + 4


		print(round(correct/total, 3))
		self.accuracy = round(correct / total, 3)
		globalVars.accuracy = round(correct / total, 3)

	# the name 'train' is taken by the parent class already
	# if report_time=True then the process func is given a second argument
	# with the amount of time the learning function has taken so far
	def learn(self, progress_func=None, report_time=False):
		start_time = time.time()
		def call_progress_func(percent):
			self.trainTime = round(time.time() - start_time, 2)
			if report_time: progress_func(percent, round(time.time() - start_time, 2))
			elif progress_func: progress_func(percent)
			else: return

		print("About to train model")
		if not image_dataset: raise IndexError('no train and test images loaded!')

		EPOCHS = self.epochs
		correct = 0
		total = 0

		train_dataset = image_dataset[0]
		test_dataset = image_dataset[1]

		loss_func = self.loss_func
		optimizer = self.optimiser_func

		print("Going through data")
		call_progress_func(9)
		total_images = EPOCHS * 60000
		for epoch in range(EPOCHS):
			self.train()
			count = epoch * 60000
			for data in train_dataset:
				# data is a batch of feature-sets(grayscale pixel values)
				# and labels(defines what number feature set corresponds to
				images, classes = data
				optimizer.zero_grad()
				output = self(images)
				loss = loss_func(output, classes)
				loss.backward()
				optimizer.step()  # Adjust weight within neural network
				call_progress_func((80 * (count / total_images)) + 10)
				# 4 images per batch (batch size is 4 in load_data())
				count = count + 4

		print('final layer bias:', getattr(self, f'layer{len(self.layer_list)-1}').bias)
		call_progress_func(90)
		print("Finding accuracy")
		count = 0
		total_images = 10000
		# Used to help fund accuracy
		with torch.no_grad():
			for data in test_dataset:
				X, y = data
				output = self(X)  # Passing in flattened image

				for idx, i in enumerate(output):
					if torch.argmax(i) == y[idx]:
						correct += 1
					total += 1
				call_progress_func(90 + (10 * count / total_images))
				count = count + 4


		print(round(correct/total, 3))
		self.accuracy = round(correct / total, 3)
		globalVars.accuracy = round(correct / total, 3)

		call_progress_func(100)
		self.trained = True
		# Save the model in cache so it can be loaded from disk next time
		torch.save(self.state_dict(), f'{cache_path}/{self.__class__.__name__}.pt')

		print("The end of training the model")

	def find_predicted_number(self):
		i = 0
		for j in range(10):
			if self.fc4.bias[j] > self.fc4.bias[i]:
				i = j
		return i


	def is_data_loaded(self):
		return bool(image_sets) and bool(image_dataset)

	# just to be consistent with is_data_loaded()
	def is_trained(self):
		return self.trained

	def remove_model_cache(self):
		rm_cache_data(self.__class__.__name__)

	def remove_all_model_cache(self):
		rm_cache_data()

	def is_model_cached(self):
		return os.path.exists(f'{cache_path}/{self.__class__.__name__}.pt')

	def load_model(self):
		print("Loading model")
		#Loading in the learnable parameters from a previously trained model
		self.load_state_dict(torch.load(f'{cache_path}/{self.__class__.__name__}.pt'))
		self.trained = True
		# lets also test the model
		self.test()

	def __str__(self):
		str = f'Model Name: {self.__class__.__name__}\n'

		if self.trained:
			str += f'Model Is Trained\n'
			if (self.accuracy):
				str += f'Model Accuracy: {self.accuracy}\n'
			else:
				str += f'Please Download The MNIST Data And Reload Model To Check Accuracy\n'
			if self.trainTime:
				str += f'Training Time: {self.trainTime}s ({int(self.trainTime) // 60 % 60}m {round(self.trainTime % 60)}s)\n'
			else:
				str += f'Model Loaded From Cache (0 Train Time)\n'
		else:
			str += f'Please train the model!\n'

		if self.modelInfo:
			str += f'Description:\n'
			str += f'    {self.modelInfo}\n'

		str += f'Internal Details:\n'
		str += f'{len(self.layer_list)} Layers/Channels:\n'
		for i, l in enumerate(self.layer_list):
			if isinstance(l, nn.Linear):
				str += f'    - #{i} (Linear): in = {l.in_features}, out = {l.out_features}\n'
			elif isinstance(l, nn.Conv2d):
				str += f'    - #{i} (Convolutional): in = {l.in_channels}, out = {l.out_channels}\n'
			elif isinstance(l, nn.MaxPool2d):
				str += f'    - #{i} (MaxPool): kernel = {l.kernel_size}, stride = {l.stride}\n'
		str += f'activation function: {self.activation_func.__class__.__name__}\n'
		str += f'# of ephocs: {self.epochs}\n'
		str += f'loss function: {self.loss_func.__class__.__name__}\n'
		str += f'optimiser: {self.optimiser_func.__class__.__name__}\n'
		str += f'learning rate: {self.learning_rate}\n'

		return str


# Define our neural network models

# 0.976 accuracy on test data
# very poor on actual hand drawn input
#  e.g. thinks a slightly long seven is a 2
class SlowLinear(NeuralModel):
	def __init__(self):
		super().__init__([
			nn.Linear(784, 100),
			nn.Linear(100, 50),
			nn.Linear(50, 50),
			nn.Linear(50, 10)
		], nn.ReLU, 10, nn.CrossEntropyLoss, optim.Adam, 0.001,
		'A slow but accurate linear model')

# inspired by https://nextjournal.com/gkoehler/pytorch-mnist
# 0.962 accuracy on test data
# not poor on actual hand drawn input
#  e.g. thinks a slightly long seven is a 3
class FastConv(NeuralModel):
	def __init__(self):
		super().__init__([
			nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
			nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
			nn.Linear(32 * 28 * 28, 50),
			nn.Linear(50, 10)
		], nn.ReLU, 3, nn.CrossEntropyLoss, optim.SGD, 0.001,
		"A fast convolutional model")

# inspired by https://nextjournal.com/gkoehler/pytorch-mnist
# 0.982 accuracy on test data
# the best by far on hand drawn input
#  e.g. recognises slightly long seven as a 7 with 82% probability (and 18% for a 2)
class SlowConv(NeuralModel):
	def __init__(self):
		super().__init__([
			nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
			nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
			nn.Linear(32 * 28 * 28, 50),
			nn.Linear(50, 10)
		], nn.ReLU, 10, nn.CrossEntropyLoss, optim.Adam, 0.001,
			"The most accurate model")

# inspired by https://nextjournal.com/gkoehler/pytorch-mnist
# 0.984 accuracy (slightly better then SlowConv) on test data
# a fair bit worse then SlowConv on actual hand drawn input, but still pretty good
#  e.g. recognises slightly long seven as 7 with 52% probability (and 14% for 2 and 4)
class SlowConvPool(NeuralModel):
	def __init__(self):
		super().__init__([
			nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
			nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
			nn.MaxPool2d(2, 2),
			nn.Linear(32 * 28 * 28, 50),
			nn.Linear(50, 10)
		], nn.ReLU, 10, nn.CrossEntropyLoss, optim.Adam, 0.001,
			"SlowConv model with a maxpool, worse then SlowConv")

# 0.986 accuracy on test data
# poor on hand drawn input
#  e.g. thinks a slightly long seven is a 2
class VerySlowConv(NeuralModel):
	def __init__(self):
		super().__init__([
			nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
			nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
			nn.MaxPool2d(2, 2),
			nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
			nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
			nn.MaxPool2d(2, 2),
			nn.Linear(128 * 28 * 28, 128),
			nn.Linear(128, 64),
			nn.Linear(64, 32),
			nn.Linear(32, 16),
			nn.Linear(16, 10),
		], nn.ReLU, 3, nn.CrossEntropyLoss, optim.Adam, 0.001,
			"A CNN with MaxPools, poor model")


# This model has many layers. but it has the lowest accuracy of the networks
# defined so far, due to over-fitting.
# 0.968 accuracy on test data
# very poor on hand drawn input
#  e.g. thinks a slightly long seven is a 3
class VerySlowConv5(NeuralModel):
	def __init__(self):
		super().__init__([
			nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
			nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
			nn.MaxPool2d(2, 2),
			nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
			nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
			nn.MaxPool2d(2, 2),
			nn.Linear(128 * 28 * 28, 128),
			nn.Linear(128, 64),
			nn.Linear(64, 32),
			nn.Linear(32, 16),
			nn.Linear(16, 10),
		], nn.ReLU, 5, nn.CrossEntropyLoss, optim.Adam, 0.001,
			"The worst model")

###### NOTE THIS MUST BE AT THE END TO WORK CORRECTLY (EXCEPT FOR DEBUG CLASS) #####
# Here we save the names of all the classes that inherit the 'NeuralModel' class
# So obviously, those subclasses must be defined before the following line
model_list = [i.__name__ for i in NeuralModel.__subclasses__()]


# Only to be used for debugging/development/program testing purposes
# IT DOES NOT TRAIN a neural network for recognition - it simply trains a rubbish network
# in the name of speed
class DebugModel(nn.Module):
	def __init__(self):
		self.trained = False
		# make these properties of the model (used by the GUI) an alias
		# to the function/global variables above
		self.download_data = download_data
		self.rm_dload_data = rm_dload_data
		self.image_sets = image_sets

		super().__init__()
		print("Training Model being prepared")

		# Define our layers
		# 28x28 = 784
		self.linear1 = nn.Linear(784, 30)
		self.out = nn.Linear(30, 10)
		self.relu = nn.ReLU()

		self.accuracy = None

	def forward(self, img):
		# Flatten the image to 784 pixels in a row
		x = img.view(-1, 784)
		x = self.relu(self.linear1(x))
		x = self.out(x)
		return x

	# the name 'train' is taken by the parent class already
	# if report_time=True then the process func is given a second argument
	# with the amount of time the learning function has taken so far
	def learn(self, progress_func=None, report_time=False):
		start_time = time.time()
		def call_progress_func(percent):
			if report_time: progress_func(percent, round(time.time() - start_time, 2))
			elif progress_func: progress_func(percent)
			else: return

		print("About to train model")
		if not image_dataset: raise IndexError('no train and test images loaded!')

		EPOCHS = 1
		correct = 0
		total = 0

		train_dataset = image_dataset[0]
		test_dataset = image_dataset[1]

		loss_func = nn.CrossEntropyLoss()
		optimizer = optim.Adam(self.parameters(), lr=0.8)

		print("Going through data")
		call_progress_func(9)
		for epoch in range(EPOCHS):
			continue
			self.train()
			for data in train_dataset:
				# data is a batch of feature-sets(grayscale pixel values)
				# and labels(defines what number feature set corresponds to
				images, classes = data
				optimizer.zero_grad()
				output = self(images.view(-1, 784))
				loss = loss_func(output, classes)
				loss.backward()
				optimizer.step()  # Adjust weight within neural network
				call_progress_func((80 * epoch / EPOCHS) + 10)

		print('final layer bias:', self.out.bias)
		call_progress_func(90)
		print("Finding accuracy")
		# Used to help fund accuracy
		with torch.no_grad():
			for data in test_dataset:
				X, y = data
				output = self(X.view(-1, 784))  # Passing in flattened image

				for idx, i in enumerate(output):
					if torch.argmax(i) == y[idx]:
						correct += 1
					total += 1


		print(round(correct/total, 3))
		self.accuracy = round(correct / total, 3)
		globalVars.accuracy = round(correct / total, 3)


		call_progress_func(100)
		self.trained = True

		print("The end of training the model")

	def find_predicted_number(self):
		i = 0
		for j in range(10):
			if self.fc4.bias[j] > self.fc4.bias[i]:
				i = j
		return i


	def is_data_loaded(self):
		return bool(image_sets) and bool(image_dataset)

	# just to be consistent with is_data_loaded()
	def is_trained(self):
		return self.trained
