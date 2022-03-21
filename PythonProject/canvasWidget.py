from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QPainter, QPen, QColor, QPixmap, QImage
from PyQt5 import QtGui
import numpy as np

DEBUG = 1  # 0 to turn DEBUG output off, 1 to turn DEBUG output on

# Build our own custom widget to provide canvas/drawing functionality
class Canvas(QWidget):
	def __init__(self, parent):
		super().__init__(parent)

		# Internal State Stuff For Our Canvas

		# Canvas width and height
		self.w = 512
		self.h = 512

		# Pen size (in radius)
		self.penWidth = 5

		# Canvas Clear Colour (bg) , and drawing colour (fg)
		self.background = 0
		self.foreground = 1

		self.resize(QSize(self.w, self.h))

		# This is the region of the canvas we need to update on the next paintEvent callback
		self.updateRect = {'x0': 0, 'y0': 0, 'x1': self.w, 'y1': self.h}

		self.lastMousePos = [None, None]

		self.pixmap = None
		self.newPixmap()

	# Override Functions

	def paintEvent(self, e):
		painter = QPainter(self)
		painter.drawPixmap(0, 0, self.w, self.h, self.pixmap)


	def drawAt(self, x0: int, y0: int) -> None:
		if self.lastMousePos[0] == None or self.lastMousePos[1] == None:
			self.lastMousePos = [x0, y0]
			return

		# update the rectangle region that has been updated/'drawn' on
		painter = QPainter(self.pixmap)
		pen = QPen()
		pen.setWidth(self.penWidth)
		pen.setColor(QColor(255 * self.foreground, 255 * self.foreground, 255 * self.foreground))
		painter.setPen(pen)
		painter.drawLine(self.lastMousePos[0], self.lastMousePos[1], x0, y0)
		self.lastMousePos = [x0, y0]

	def mouseMoveEvent(self, e: QtGui.QMouseEvent) -> None:
		x = int(e.localPos().x())
		y = int(e.localPos().y())

		if x < 0 or x > self.w:  # mouse dragged outside of drawing area
			if DEBUG: print("Mouse dragged outside of drawing area (on x-axis) !")
			return
		if y < 0 or y > self.h: # same as above
			if DEBUG: print("Mouse dragged outside of drawing area (on y-axis) !")
			return

		# 'paint' on the canvas
		self.drawAt(x, y)
		# Fire the Qt redraw event to update the canvas widget
		self.update()

	def mouseReleaseEvent(self, e: QtGui.QMouseEvent) -> None:
		self.lastMousePos = [None, None]
		if DEBUG:
			self.canvas2image()

	def resizeEvent(self, e: QtGui.QResizeEvent) -> None:
		self.w = e.size().width()
		self.h = e.size().height()

		if DEBUG:
			print(f'New Canvas Size: ({self.w}, {self.h})')

		self.newPixmap()

	# End of overridden functions

	def newPixmap(self):
		self.pixmap = QPixmap(self.w, self.h)
		self.pixmap.fill(QColor(self.background * 255, self.background * 255, self.background * 255))

	def clearCanvas(self):
		self.newPixmap()
		self.update()  # Draw our new pixmap

	def canvas2image(self) -> np.ndarray :
		# Here we must convert our QPixmap to a numpy array
		# First convert the pixmap to an image
		img = self.pixmap.toImage()
		bitdepth = img.depth() // 8  # we want it in bytes, img.depth() is in bits
		# get a pointer to the data of the first pixel
		img_array = img.bits()
		# Fill out the "size" of the image so we know how many bytes
		# beyond the pointer the image makes up
		img_array.setsize(self.w * self.h * bitdepth)
		# Now create a numpy array from the raw bytes we have above
		# (tell numpy the image data is in bytes)
		#
		# Remember matrix order is (Rows (height) , Cols (Width) )
		nparray = np.array(img_array, np.uint8).reshape((self.h, self.w, bitdepth))
		# turn it into a monochrome image (with values between 0..1)
		monoarray = np.zeros((self.h, self.w))
		for i in range(self.h):
			for j in range(self.w):
				# check if all colour channels are more then 0
				# note in our case since we only draw in black and white,
				# if one channel is > 0 then all should be
				if (nparray[i][j][0:3] > 0).all():
					monoarray[i][j] = 255 * self.foreground
				else:
					monoarray[i][j] = 255 * self.background

		return monoarray
