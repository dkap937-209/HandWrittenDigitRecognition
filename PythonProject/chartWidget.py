from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import QSize, QRect, QPoint
from PyQt5.QtGui import QPainter, QColor, QPixmap
from PyQt5 import QtGui
import globalVars

# Build our own custom widget to provide a graph for results
class Chart(QWidget):
	def __init__(self, parent):
		super().__init__(parent)

		# Internal State Stuff For Our Canvas

		# Canvas width and height
		self.w = 512
		self.h = 512

		# Number of pixels to leave as padding around edge of canvas and
		# between each bar
		self.padding = 10
		self.background = QColor(255, 255, 255)

		self.resize(QSize(self.w, self.h))
		self.pixmap = QPixmap(self.w, self.h)

	# Override Functions

	def paintEvent(self, e):
		painter = QPainter(self)
		painter.fillRect(QRect(0, 0, self.w, self.h), self.background)

		if globalVars.predictionResults is None:
			return

		# Number of pixels to leave as padding around edge of canvas and
		# effective width and height (after removing padding)
		self.effective_w = self.w - (self.padding * 2)
		self.effective_h = self.h - (self.padding * 2)

		# Length of one percent
		# We want the largest probability to take up most of the canvas
		# (with enough padding to write 100% at the end of the bar)
		self.length_per_step = (self.effective_w - painter.fontMetrics().width("100%  ")) / (float(max(globalVars.predictionResults)) * 100)
		# need spaces for 10 classes (10 digits)
		self.thickness = (self.effective_h - self.padding * 9) / 10

		for index, value in enumerate(globalVars.predictionResults):
			x0 = self.padding
			# self.padding because one time for starting point of graph
			# and add it a second time for the sapce between the last bar
			# and this bar (unless we are drawing the very first bar)
			y0 = self.padding + (self.thickness * index) + (self.padding * index)

			x1 = self.length_per_step * (float(value) * 100)
			y1 = y0 + self.thickness

			font_width = painter.fontMetrics().width(str(index))
			font_height = painter.fontMetrics().height()

			# center label w.r.t. the bar vertically
			painter.drawText(QPoint(x0, (y0 + (y1 - y0) / 2) + font_height / 2), str(index))
			# Add about ten  pixels for the class label text
			painter.fillRect(QRect(x0 + font_width + 5, y0, x1, self.thickness), QColor(0, 0, 0))
			# center percentage w.r.t. the bar vertically, and put some
			# space between the end of the bar and the percentage text
			painter.drawText(QPoint(x1 + font_width + 17, (y0 + (y1 - y0) / 2) + font_height / 2), str(round(float(value) * 100)) + '%')


	def resizeEvent(self, e: QtGui.QResizeEvent) -> None:
		self.w = e.size().width()
		self.h = e.size().height()
		self.update()
