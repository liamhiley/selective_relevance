#!/usr/bin/env python3

import sys
from PyQt5.QtWidgets import QWidget, QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QGridLayout, QShortcut
from PyQt5.QtGui import QPixmap, QImage, QColor, QPainter, QPen, QCursor, QKeySequence
from PyQt5.QtCore import Qt, QRect, QPoint, QSize

class AppWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        top, left, width, height = 400,400,800,600
        self.setGeometry(top, left, width, height)
        self.setWindowTitle("Relevance Painter")
        self.setCentralWidget(RelevancePainter((top,left,width,height)))

class RelevancePainter(QWidget):
    def __init__(self,geometry):
        super().__init__()
        # set layout of painter

        layout = QGridLayout()
        self.geometry = geometry
        self.spacing = 10
        layout.setSpacing(self.spacing)
        canvas = RelevanceCanvas("me_and_gimli.jpg",geometry+(self.spacing,))
        layout.addWidget(canvas,1,0,3,3)
        self.setLayout(layout)

class RelevanceCanvas(QWidget):
    def __init__(self, img,geometry):
        super().__init__()

        # We need to update the widgets geometry to account for it being nested in a main window
        top,left,width,height,spacing = geometry
        top += spacing
        left += spacing
        width -= spacing
        height -= spacing
        self.setGeometry(top, left, width, height)

        # set the background to the initial frame
        self.image = QImage(img)
        # set opaque overlay for frame
        self.drawOverlay = QImage(self.size(),QImage.Format_Indexed8)
        self.drawOverlay.setColorTable([Qt.transparent, QColor(0,0,0,127).rgb()])
        self.drawOverlay.fill(QColor(0,0,0,127))

        self.drawing = False

        # default brushes
        self._clear_size = 60
        # draw
        # self.drawBrush =

        # set keybindings for reveal and replace modes
        self.shortcut = QShortcut(QKeySequence("D"), self)
        self.shortcut.activated.connect(self.setDrawCursor)
        self.shortcut = QShortcut(QKeySequence("E"), self)
        self.shortcut.activated.connect(self.setEraseCursor)

    def paintEvent(self,event):
        canvasPainter = QPainter(self)
        canvasPainter.drawImage(self.rect(),self.image,self.image.rect())
        canvasPainter.drawImage(self.rect(),self.drawOverlay,self.drawOverlay.rect())

    def mousePressEvent(self,event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.lastPoint = event.pos()

    def mouseMoveEvent(self,event):
        if event.buttons() and Qt.LeftButton and self.drawing:
            painter = QPainter(self.drawOverlay)
            # painter.QPen(QColor(0,0,0,127))
            rect = QRect(QPoint(), self._clear_size*QSize())
            rect.moveCenter(event.pos())
            bg = self.image
            if self.drawing:
                # get rectangle positions
                for y in range(rect.bottom(),rect.top()):
                    for x in range(rect.right(),rect.left()):
                        colour_at_pos = bg.pixelColor(x,y)
                        # print(colour_at_pos)
                        painter.setPen(QPen(colour_at_pos))
                        painter.drawPoint(x,y)
                # painter.drawRect(rect)

            else:
                # painter.save()
                # painter.setCompositionMode(QPainter.CompositionMode_Clear)
                painter.eraseRect(rect)
                # painter.fillRect(rect,QColor(0,0,0,127))
                # painter.restore()
            self.lastPoint = self.mapToGlobal(event.pos())
            self.update()
            painter.end()


    def mouseReleaseEvent(self,event):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def setDrawCursor(self):
        self.drawing = True
        self.setCursor()
    def setEraseCursor(self):
        self.drawing = False
        self.setCursor()

    def setCursor(self):
        pixmap = QPixmap(QSize(1,1)*self._clear_size)
        if self.drawing:
            pixmap.fill(QColor(0,0,0,127))
        else:
            pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setPen(QPen(Qt.black,2))
        painter.drawRect(pixmap.rect())
        painter.end()
        cursor = QCursor(pixmap)
        QApplication.setOverrideCursor(cursor)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = AppWindow()
    window.show()

    app.exec_()
