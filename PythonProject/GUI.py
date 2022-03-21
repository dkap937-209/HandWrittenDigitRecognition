import numpy
from PyQt5 import QtCore, QtGui, QtWidgets
from canvasWidget import Canvas
from chartWidget import Chart
from neuralNetwork import rm_dload_data, model_list
import globalVars


# Because Qt does not provide a way to center a window, we can do the maths ourself
def center_on_screen(application: QtWidgets.QApplication, window: QtWidgets.QWidget):
	# Take into account window manager decorations with 'availableGeometry'
	screenWidth = application.desktop().availableGeometry().width()
	screenHeight = application.desktop().availableGeometry().height()

	winWidth = window.width()
	winHeight = window.height()

	window.move((screenWidth - winWidth) // 2, (screenHeight - winHeight) // 2)


class GUI:
	# - model is the neural network model instance
	# - active_model_func is the function that gets called when the user selects a model
	# - process_func is the function that gets called when the 'recognise' button is pressed
	def __init__(self, active_model_func, process_func):
		# Store a reference to the image processing function
		self.active_model_func = active_model_func
		self.process_func = process_func
		# self.model = None  # initially we have no model
		self.app = QtWidgets.QApplication([""])
		self.window = QtWidgets.QMainWindow()
		self.setupUi()
		self.window.show()
		self.app.exec()

	def attachModel(self):
		# Reset all models to their unchecked state:
		for i in model_list:
			getattr(self, i).setText(i)
		self.DebugModel.setText("__DEBUG__")

		# Set the model info
		self.ModelInfoOutput.setText("<b>Model Info:</b>")


		if globalVars.model is None:
			self.ModelInfoOutput.append("<i>Please select a model.</i>")
			return
		elif globalVars.model.__class__.__name__ == "DebugModel":
			self.ModelInfoOutput.append("<b><i><u>!!! WARNING: THIS MODEL IS ONLY FOR PROGRAM DEVELOPMENT AND DEBUGGING !!!</u></i></b>")
			self.DebugModel.setText(f'__DEBUG__    ✓')
		else:
			self.ModelInfoOutput.append(str(globalVars.model))
			x = globalVars.model.__class__.__name__
			getattr(self, x).setText(f'{x}    ✓')

		# If the box is to small, Qt will scroll the text box down
		# undo that - we want to show the user the info @ the top first
		self.ModelInfoOutput.verticalScrollBar().triggerAction(QtWidgets.QScrollBar.SliderToMinimum)

	def setupUi(self):
		# Create the main application window
		self.window.setObjectName("MainWindow")
		self.window.setWindowTitle("Handwritten Digit Recogniser")
		self.window.resize(600, 400)
		self.window.setMinimumSize(600, 400)
		center_on_screen(self.app, self.window)

		self.WindowContent = QtWidgets.QWidget(self.window)
		self.WindowContent.setObjectName("WindowContent")
		self.window.setCentralWidget(self.WindowContent)

		# menu / status bar code ...

		self.Menubar = QtWidgets.QMenuBar(self.window)
		self.Menubar.setGeometry(QtCore.QRect(0, 0, 919, 30))
		self.Menubar.setObjectName("Menubar")

		# File Menu
		self.menuFile = QtWidgets.QMenu(self.Menubar)
		self.menuFile.setObjectName("menuFile")
		self.menuFile.setTitle("File")

		# Creating train model menu option
		self.actionTrainModel = QtWidgets.QAction(self.window)
		self.actionTrainModel.setObjectName("actionTrainModel")
		QtCore.QMetaObject.connectSlotsByName(self.window)
		self.actionTrainModel.triggered.connect(self.setupTrainUI)
		self.actionTrainModel.setText("Train Model")
		self.actionTrainModel.setShortcut("Ctrl+T")

		# Creating remove downloaded data menu option
		self.actionRemoveData = QtWidgets.QAction(self.window)
		self.actionRemoveData.triggered.connect(rm_dload_data)
		self.actionRemoveData.setText("Delete Downloaded Data")
		self.actionRemoveData.setShortcut("CTRL+D")

		# Creating load model menu option
		self.actionLoad = QtWidgets.QAction(self.window)
		self.actionLoad.triggered.connect(self.loadTrainedModel)
		self.actionLoad.setText("Load Model From Cache")
		self.actionLoad.setShortcut("CTRL+L")

		# Creating Delete Current Model Cache option
		self.actionRmCurrentCache = QtWidgets.QAction(self.window)
		self.actionRmCurrentCache.triggered.connect(self.removeCurrentModelCache)
		self.actionRmCurrentCache.setText("Remove Model Cache")
		self.actionRmCurrentCache.setShortcut("ALT+D")

		# Creating Delete all Model Cache option
		self.actionRmAllCache = QtWidgets.QAction(self.window)
		self.actionRmAllCache.triggered.connect(self.removeAllModelCache)
		self.actionRmAllCache.setText("Remove All Model Cache")
		self.actionRmAllCache.setShortcut("CTRL+ALT+D")

		# Creating Quit menu option
		self.actionQuit = QtWidgets.QAction(self.window)
		self.actionQuit.setObjectName("actionQuit")
		self.actionQuit.setShortcut("Ctrl+Q")
		self.actionQuit.triggered.connect(self.window.close)
		self.actionQuit.setText("Quit")
		self.menuFile.addSeparator()

		# Add actions to file menu
		self.menuFile.addAction(self.actionTrainModel)
		self.menuFile.addAction(self.actionRemoveData)
		self.menuFile.addAction(self.actionLoad)
		self.menuFile.addAction(self.actionRmCurrentCache)
		self.menuFile.addAction(self.actionRmAllCache)
		self.menuFile.addAction(self.actionQuit)

		# View Menu
		self.menuView = QtWidgets.QMenu(self.Menubar)
		self.menuView.setObjectName("menuView")
		self.menuView.setTitle("View")
		self.menuView.addSeparator()

		self.window.setMenuBar(self.Menubar)

		# Adding option to view training images
		self.actionViewTrainingImg = QtWidgets.QAction(self.window)
		self.actionViewTrainingImg.setText("View Training Images")
		self.actionViewTrainingImg.setObjectName("actionViewTrainingImg")
		self.actionViewTrainingImg.triggered.connect(lambda: self.setupViewImages('Train'))
		self.actionViewTrainingImg.setShortcut("Ctrl+Alt+T")

		# Adding option to view testing images
		self.actionViewTestingImg = QtWidgets.QAction(self.window)
		self.actionViewTestingImg.setObjectName("actionViewTestingImg")
		self.actionViewTestingImg.setText("View Testing Images")
		self.actionViewTestingImg.triggered.connect(lambda: self.setupViewImages('Test'))
		self.actionViewTestingImg.setShortcut("Ctrl+Shift+T")

		# Model Selection Menu
		self.menuModelSel = QtWidgets.QMenu(self.Menubar)
		self.menuModelSel.setObjectName("menuMOdelSel")
		self.menuModelSel.setTitle("Model Select")

		# Adding Model Options
		# Generate a callback function to switch to this current (i-th) model
		# and then return that function. (Lambdas do not work in this dynamic loop)
		def gen_model_callback(name):
			def func():
				self.active_model_func(self.attachModel, name)
			return func
		for n, i in enumerate(model_list):
			setattr(self, i, QtWidgets.QAction(self.window))
			getattr(self, i).setObjectName(f'action{i}')
			getattr(self, i).setText(i)
			getattr(self, i).triggered.connect(gen_model_callback(i))
			# Set the shortcut
			getattr(self, i).setShortcut(f"Alt+{n}")

		# Adding option to use DEBUG model
		self.DebugModel = QtWidgets.QAction(self.window)
		self.DebugModel.setObjectName("actionDebugModel")
		self.DebugModel.setText("__DEBUG__")
		self.DebugModel.triggered.connect(lambda: self.active_model_func(self.attachModel, 'DebugModel'))

		# Add items to Menus
		self.Menubar.addAction(self.menuFile.menuAction())
		self.Menubar.addAction(self.menuView.menuAction())
		self.Menubar.addAction(self.menuModelSel.menuAction())
		self.menuView.addAction(self.actionViewTrainingImg)
		self.menuView.addAction(self.actionViewTestingImg)
		# Add in model options
		for i in model_list:
			self.menuModelSel.addAction(getattr(self, i))
		self.menuModelSel.addAction(self.DebugModel)

		# Main window layout ...

		# Creating grid layout for the window
		self.gridLayout = QtWidgets.QGridLayout(self.WindowContent)
		self.gridLayout.setObjectName("gridLayout")
		self.VerticalDivider = QtWidgets.QHBoxLayout()
		self.VerticalDivider.setObjectName("VerticalDivider")

		# Create vertical divider to split grid layout into two halfs
		# One half for drawing canvas, other half for buttons etc.
		self.RightHalf = QtWidgets.QVBoxLayout()
		self.RightHalf.setObjectName("ButtonLayout")

		# Create canvas frame to hold canvas drawing widget
		# (Left side of vertical divider)
		self.DrawingCanvas = QtWidgets.QFrame(self.WindowContent)
		self.DrawingCanvas.setLayout(QtWidgets.QVBoxLayout())
		self.DrawingCanvas.setFrameShape(QtWidgets.QFrame.StyledPanel)
		self.DrawingCanvas.setFrameShadow(QtWidgets.QFrame.Raised)
		self.DrawingCanvas.setObjectName("DrawingCanvas")
		self.canvas = Canvas(self.DrawingCanvas)
		# So that when the DrawingCanvas is resized, the canvas is also resized
		self.DrawingCanvas.layout().addWidget(self.canvas)

		# Creating area for displaying information about the current model
		self.ModelInfoOutput = QtWidgets.QTextEdit(self.DrawingCanvas)
		self.ModelInfoOutput.setReadOnly(True)
		self.ModelInfoOutput.setMaximumHeight(150)

		self.ModelInfoOutput.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
		self.DrawingCanvas.layout().addWidget(self.ModelInfoOutput)
		# This will init the content of the model info box
		self.attachModel()

		# Buttons etc for right side of vertical divider

		# Button to clear drawing canvas
		self.ClearButton = QtWidgets.QPushButton(self.WindowContent)
		self.ClearButton.setObjectName("ClearButton")
		self.ClearButton.clicked.connect(self.canvas.clearCanvas)
		# note the '&' sets the next character as a shortcut (ALT + c)
		self.ClearButton.setText("&Clear")

		# Creating button to recognise image drawn on the canvas
		def recognise_func():
			if globalVars.model and globalVars.model.is_trained():
				self.process_func(self.canvas.canvas2image())
				self.updateNum()
				self.graph.update()
			else:
				msgBox = QtWidgets.QMessageBox()
				msgBox.setWindowTitle("Recognise Digit")
				msgBox.setText("Please select and/or train the model first to view images")
				msgBox.exec()

		self.RecogniseButton = QtWidgets.QPushButton(self.WindowContent)
		self.RecogniseButton.setObjectName("RecogniseButton")
		self.RecogniseButton.clicked.connect(recognise_func)
		# note the '&' sets the next character as a shortcut (ALT + r)
		self.RecogniseButton.setText("&Recognise")

		# Label under clear/reset buttons for class probability etc.
		self.ClassProbabilityLabel = QtWidgets.QLabel(self.WindowContent)
		self.ClassProbabilityLabel.setObjectName("ClassProbabilityLabel")
		self.ClassProbabilityLabel.setText("Class Probability:")

		# Create QFrame to hold class graph
		self.ClassProbabilityGraph = QtWidgets.QFrame(self.WindowContent)
		self.ClassProbabilityGraph.setFrameShape(QtWidgets.QFrame.StyledPanel)
		self.ClassProbabilityGraph.setFrameShadow(QtWidgets.QFrame.Raised)
		self.ClassProbabilityGraph.setObjectName("ClassProbabilityGraph")
		self.ClassProbabilityGraph.setLayout(QtWidgets.QHBoxLayout())
		self.ClassProbabilityGraph.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
		self.graph = Chart(self.ClassProbabilityGraph)
		self.ClassProbabilityGraph.layout().addWidget(self.graph)

		# Label for detected number
		self.NumberLabel = QtWidgets.QLabel(self.WindowContent)
		self.NumberLabel.setObjectName("NumberLabel")
		self.NumberLabel.setText("<b>Number Recognized:</b>")

		# Add all the created widgets into their respective layouts now
		self.VerticalDivider.addWidget(self.DrawingCanvas)
		self.RightHalf.addWidget(self.ClearButton)
		self.RightHalf.addWidget(self.RecogniseButton)
		self.RightHalf.addWidget(self.ClassProbabilityLabel)
		self.RightHalf.addWidget(self.ClassProbabilityGraph)
		self.RightHalf.addWidget(self.NumberLabel)
		self.VerticalDivider.addLayout(self.RightHalf)
		self.gridLayout.addLayout(self.VerticalDivider, 0, 0, 1, 1)

	# Update the recognised class and class probabilities
	def updateNum(self):
		self.NumberLabel.setText(f'<b>Number Recognized: {globalVars.predictedNum}</b>')

	def setupTrainUI(self):
		# print(self.model)
		# if self.model is None:
		if globalVars.model is None:
			msgBox = QtWidgets.QMessageBox()
			msgBox.setWindowTitle("Select Model")
			msgBox.setText("Please select a model to use first")
			msgBox.exec_()
		else:
			self.trainWindow = QtWidgets.QMainWindow()
			self.trainWindow.setObjectName("MainWindow")
			self.trainWindow.resize(770, 408)
			# self.trainWindow.setEnabled(770, 408)
			self.trainWindow.setWindowTitle("Train Model")
			self.trainWindow.setFixedSize(770, 408)

			self.trainWindowCentralwidget = QtWidgets.QWidget(self.trainWindow)
			self.trainWindowCentralwidget.setObjectName("centralwidget")

			# Creating output area for when downloading MNIST data set and training
			self.trainWindowTxtOutput = QtWidgets.QTextEdit(self.trainWindowCentralwidget)
			self.trainWindowTxtOutput.setGeometry(QtCore.QRect(50, 50, 671, 111))
			self.trainWindowTxtOutput.setReadOnly(True)

			# Train Progress Bar
			self.trainWindowTrainProgressBar = QtWidgets.QProgressBar(self.trainWindowCentralwidget)
			# self.trainWindowTrainProgressBar.setGeometry(QtCore.QRect(50, 190, 681, 23))
			self.trainWindowTrainProgressBar.setGeometry(QtCore.QRect(200, 220, 531, 23))
			self.trainWindowTrainProgressBar.setProperty("value", 0)
			self.trainWindowTrainProgressBar.setObjectName("progressBar")

			# Download Progress Bar
			self.trainWindowDloadProgressBar = QtWidgets.QProgressBar(self.trainWindowCentralwidget)
			self.trainWindowDloadProgressBar.setGeometry(QtCore.QRect(200, 190, 531, 23))
			self.trainWindowDloadProgressBar.setProperty("value", 0)
			self.trainWindowDloadProgressBar.setObjectName("progressBar")

			# Download button for MNIST dataset
			self.trainWindowBtnDownload = QtWidgets.QPushButton(self.trainWindowCentralwidget)
			self.trainWindowBtnDownload.setGeometry(QtCore.QRect(50, 190, 121, 23))
			self.trainWindowBtnDownload.setObjectName("btnDownload")
			self.trainWindowBtnDownload.setText("Download MNIST")

			# if not self.model.is_data_loaded():
			if not globalVars.model.is_data_loaded():
				self.trainWindowBtnDownload.clicked.connect(lambda: (self.trainWindowBtnDownload.setEnabled(False), globalVars.model.download_data(downloadProgress)))
			else:
				# Disable button if data is already loaded & set progress bar to 100%
				self.trainWindowTxtOutput.setText("MNIST dataset already loaded")
				self.trainWindowDloadProgressBar.setValue(100)
				self.trainWindowBtnDownload.setEnabled(False)

			def setTrainedTxt():
				self.trainWindowTxtOutput.append("Trained")
				self.trainWindowTxtOutput.append("Accuracy " + str(globalVars.accuracy * 100) + "%")

			def updateTrainProgressBar():
				self.trainWindowTrainProgressBar.setValue(globalVars.progressBarVal)

			# Button to train the model
			# We only want to show this button if the MNIST data has been downloaded
			def createTrainBtn():
				def trainingCallback(progress, elapsed_time):
					if progress == 9:
						self.trainWindowBtnTrain.setEnabled(False)
						self.trainWindowTxtOutput.append("Starting Training")
					elif progress == 90:
						self.trainWindowTxtOutput.append("Training Finished")
						self.trainWindowTxtOutput.append("Testing Model")
					elif progress == 100:
						if elapsed_time > 59:
							minutes = int(elapsed_time) // 60 % 60
							seconds = round(elapsed_time % 60)
							self.trainWindowTxtOutput.append(f"Model Trained in {elapsed_time} seconds ({minutes}m {seconds}s)")
						else:
							self.trainWindowTxtOutput.append(f"Model Trained in {elapsed_time} seconds")
						self.trainWindowTxtOutput.append("Accuracy " + str(round(globalVars.accuracy * 100, 2)) + "%")
					self.trainWindowTrainProgressBar.setValue(progress)
					# Give Qt a change to process thing so the Operating
					# System doesn't think the application has hung
					self.app.processEvents()

				self.trainWindowBtnTrain = QtWidgets.QPushButton(self.trainWindowCentralwidget)
				self.trainWindowBtnTrain.setGeometry(QtCore.QRect(50, 220, 121, 23))
				self.trainWindowBtnTrain.setObjectName("btnTrain")
				self.trainWindowBtnTrain.setText("Train")
				if not globalVars.model.is_trained():
					self.trainWindowBtnTrain.clicked.connect(lambda: globalVars.model.learn(trainingCallback, True))
				else:
					# Disable button if model is already trained & set progress bar to 100%
					self.trainWindowTxtOutput.append("Default Neural Network Already Trained")
					self.trainWindowTrainProgressBar.setValue(100)
					self.trainWindowBtnTrain.setEnabled(False)
				self.trainWindowBtnTrain.show()

			# Update the progress of the mnist file download
			def downloadProgress(percent):
				if percent == 5:
					self.trainWindowTxtOutput.append('Starting Download....')
				elif percent == 90:
					self.trainWindowTxtOutput.append('Download Complete.....')
					self.trainWindowTxtOutput.append('Unpacking Files......')
				elif percent == 100:
					self.trainWindowTxtOutput.append('Done! Ready to train model.......')
					# The entire download process is complete,
					# let's enable the train button for the user
					createTrainBtn()
				self.trainWindowDloadProgressBar.setValue(percent)
				# Give Qt a change to process thing so the Operating
				# System doesn't think the application has hung
				self.app.processEvents()

			# if self.model.is_data_loaded(): createTrainBtn()
			if globalVars.model.is_data_loaded(): createTrainBtn()

			# # To close the window
			# self.trainWindowBtnCancel = QtWidgets.QPushButton(self.trainWindowCentralwidget)
			# self.trainWindowBtnCancel.setGeometry(QtCore.QRect(560, 260, 121, 28))
			# self.trainWindowBtnCancel.setObjectName("btnCancel")
			# self.trainWindowBtnCancel.setText("Continue")
			# self.trainWindowBtnCancel.clicked.connect(self.trainWindow.close)

			self.trainWindow.setCentralWidget(self.trainWindowCentralwidget)

			QtCore.QMetaObject.connectSlotsByName(self.trainWindow)

			# Menu Bar and status bar
			self.trainWindowMenubar = QtWidgets.QMenuBar(self.trainWindow)
			self.trainWindowMenubar.setGeometry(QtCore.QRect(0, 0, 770, 26))
			self.trainWindowMenubar.setObjectName("menubar")
			self.trainWindow.setMenuBar(self.trainWindowMenubar)
			self.trainWindowStatusbar = QtWidgets.QStatusBar(self.trainWindow)
			self.trainWindowStatusbar.setObjectName("statusbar")
			self.trainWindow.setStatusBar(self.trainWindowStatusbar)

			center_on_screen(self.app, self.trainWindow)
			self.trainWindow.show()


	def loadTrainedModel(self):
		if globalVars.model is not None and globalVars.model.is_model_cached():
			globalVars.model.load_model()
			# Refill the model info box
			self.attachModel()
		else:
			msgBox = QtWidgets.QMessageBox()
			msgBox.setWindowTitle("Loading Trained Model")
			msgBox.setText("No Trained Model Exist")
			msgBox.exec()

	def removeCurrentModelCache(self):
		if globalVars.model is not None and globalVars.model.is_model_cached():
			globalVars.model.remove_model_cache()
		else:
			msgBox = QtWidgets.QMessageBox()
			msgBox.setWindowTitle("Loading Trained Model")
			msgBox.setText("No Trained Model Exist")
			msgBox.exec()

	def removeAllModelCache(self):
		if globalVars.model is not None:
			globalVars.model.remove_all_model_cache()
		else:
			msgBox = QtWidgets.QMessageBox()
			msgBox.setWindowTitle("Loading Trained Model")
			msgBox.setText("No Trained Model Exist")
			msgBox.exec()

	def setupViewImages(self, mode):
		def showImages():
			boxSize = 30  # image boxes are 30*30 pixels
			boxSpacingX = 25  # space between each image box horizontally
			boxSpacingY = 20  # space between each image box vertically

			boxStartOffset = 80  # The 'previous' button ends here (horizontally)
			# we are creating the Qt image labels for the first time (window has just been opened)
			if not self.viewImagesBoxes:
				for i in range(0, 11):
					for j in range(0, 11):
						imgNo = (self.viewImagesPage * self.viewImagesImagesPerPage) + (i * 11 + j)
						# convert the image to a numpy array, # scale the numpy values from {0.f .. 1.f}
						# to {0.f .. 255.f} and then # convert from float to uint8 (byte)
						image = (self.viewImagesDataset[imgNo][0].numpy()[0] * 255).astype(numpy.uint8)
						# (y = mx + c)
						x0 = boxStartOffset + ((boxSize + boxSpacingX) * j + boxSpacingX)
						y0 = ((boxSize + boxSpacingY) * i + boxSpacingY)

						# Create the space for the image
						imgbox = QtWidgets.QLabel(self.viewImagesCentralWidget)
						imgbox.setGeometry(QtCore.QRect(x0, y0, boxSize, boxSize))
						imgbox.setFont(font)
						imgbox.setText("[]")
						imgbox.setPixmap(QtGui.QPixmap(QtGui.QImage(image, 28, 28, 28, QtGui.QImage.Format_Grayscale8)))
						self.viewImagesBoxes.append([imgNo, imgbox])
						print(f'viewing image: {imgNo}')
			# we are replacing the existing image labels (user hit 'next' button)
			else:
				for i in range(0, 11):
					for j in range(0, 11):
						imgIdx = i * 11 + j
						imgNo = (self.viewImagesPage * self.viewImagesImagesPerPage) + (i * 11 + j)
						# convert the image to a numpy array, # scale the numpy values from {0.f .. 1.f}
						# to {0.f .. 255.f} and then # convert from float to uint8 (byte)
						try:
							image = (self.viewImagesDataset[imgNo][0].numpy()[0] * 255).astype(numpy.uint8)
						except IndexError:  # We've reached the end of the dataset, nothing more to display
							imgNo = None
							image = None
							print('\t ----- \t')

						# place the image into the box it belongs in
						self.viewImagesBoxes[imgIdx][0] = imgNo
						self.viewImagesBoxes[imgIdx][1].setPixmap(
							QtGui.QPixmap(QtGui.QImage(image, 28, 28, 28, QtGui.QImage.Format_Grayscale8)))
						print(f'viewing image: {imgNo}')
			print('\t ----- \t')

		def nextPage():
			if self.viewImagesPage + 1 > self.viewImagesMaxpages:
				return
			self.viewImagesPage = self.viewImagesPage + 1
			self.viewImagesPageEdit.setText(f'{self.viewImagesPage}')
			showImages()
			# because on the last page the last item in the list is not the last valid image
			lastIdx = [i for i, e in enumerate(self.viewImagesBoxes) if e[0] is not None][-1]
			self.viewImagesPageLabel.setText(
				f' / {self.viewImagesMaxpages} (Viewing Images {self.viewImagesBoxes[0][0]} - {self.viewImagesBoxes[lastIdx][0]})')

		def prevPage():
			if self.viewImagesPage - 1 < 0:
				return
			self.viewImagesPage = self.viewImagesPage - 1
			self.viewImagesPageEdit.setText(f'{self.viewImagesPage}')
			showImages()
			# because on the last page the last item in the list is not the last valid image
			lastIdx = [i for i, e in enumerate(self.viewImagesBoxes) if e[0] is not None][-1]
			self.viewImagesPageLabel.setText(
				f' / {self.viewImagesMaxpages} (Viewing Images {self.viewImagesBoxes[0][0]} - {self.viewImagesBoxes[lastIdx][0]})')

		def setPage():
			# input is already validated by the Qt Widget
			self.viewImagesPage = int(self.viewImagesPageEdit.text())
			showImages()
			# because on the last page the last item in the list is not the last valid image
			lastIdx = [i for i, e in enumerate(self.viewImagesBoxes) if e[0] is not None][-1]
			self.viewImagesPageLabel.setText(
				f' / {self.viewImagesMaxpages} (Viewing Images {self.viewImagesBoxes[0][0]} - {self.viewImagesBoxes[lastIdx][0]})')

		try:
			if mode == "Test":
				# imageDataset = self.model.image_sets[1]
				imageDataset = globalVars.model.image_sets[1]
				title = "View Test images"
			else:
				imageDataset = globalVars.model.image_sets[0]
				title = "View Training images"
		except:
			imageDataset = None

		if not imageDataset:
			msgBox = QtWidgets.QMessageBox()
			msgBox.setWindowTitle("View Training and Testing Images")
			msgBox.setText("Please select the model first to view images")
			msgBox.exec()
			return

		print('there is data')
		self.viewImagesDataset = imageDataset
		# we have 11x11 images is 121 images per page
		self.viewImagesImagesPerPage = 121
		self.viewImagesMaxpages = (len(self.viewImagesDataset) // self.viewImagesImagesPerPage)
		self.viewImagesBoxes = []
		self.viewImagesPage = 0

		self.viewImagesWindow = QtWidgets.QMainWindow()
		self.viewImagesWindow.setObjectName("viewImages")
		self.viewImagesWindow.setWindowTitle(title)
		self.viewImagesWindow.setFixedSize(800, 600)

		self.viewImagesCentralWidget = QtWidgets.QWidget(self.viewImagesWindow)
		self.viewImagesCentralWidget.setObjectName("centralwidget")

		# Setting font for buttons
		font = QtGui.QFont()
		font.setPointSize(36)
		font.setBold(True)
		font.setWeight(75)

		# Previous Page Button
		self.viewImagesBtnLeft = QtWidgets.QPushButton(self.viewImagesCentralWidget)
		self.viewImagesBtnLeft.setGeometry(QtCore.QRect(20, 270, 60, 60))
		self.viewImagesBtnLeft.setFont(font)
		self.viewImagesBtnLeft.setText("<")
		self.viewImagesBtnLeft.clicked.connect(prevPage)

		# Next page button
		self.viewImagesBtnRight = QtWidgets.QPushButton(self.viewImagesCentralWidget)
		self.viewImagesBtnRight.setGeometry(QtCore.QRect(720, 270, 60, 60))
		self.viewImagesBtnRight.setFont(font)
		self.viewImagesBtnRight.setText(">")
		self.viewImagesBtnRight.clicked.connect(nextPage)

		# Now display the images
		showImages()
		# Page number details
		# Page number input
		self.viewImagesPageEdit = QtWidgets.QLineEdit(self.viewImagesCentralWidget)
		self.viewImagesPageEdit.setGeometry(QtCore.QRect(300, 560, 30, 30))
		self.viewImagesPageEdit.setText(f'{self.viewImagesPage}')
		self.viewImagesPageEdit.setValidator(QtGui.QIntValidator(0, len(self.viewImagesDataset) // 121))
		self.viewImagesPageEdit.editingFinished.connect(setPage)
		# static label
		self.viewImagesPageLabel = QtWidgets.QLabel(self.viewImagesCentralWidget)
		self.viewImagesPageLabel.setGeometry(QtCore.QRect(330, 560, 250, 30))

		self.viewImagesPageLabel.setText(
			f' / {self.viewImagesMaxpages} (Viewing Images {self.viewImagesBoxes[0][0]} - {self.viewImagesBoxes[-1][0]})')

		self.viewImagesWindow.setCentralWidget(self.viewImagesCentralWidget)
		center_on_screen(self.app, self.viewImagesWindow)
		self.viewImagesWindow.show()
