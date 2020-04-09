#!/usr/bin/env python3

import sys
import pdb

from PyQt5.QtWidgets import QWidget, QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QGridLayout, QShortcut, QPushButton, QFileDialog
from PyQt5.QtGui import QPixmap, QImage, QColor, QPainter, QPen, QCursor, QKeySequence, QPalette
from PyQt5.QtCore import Qt, QRect, QPoint, QSize

import video
import cv2

class AppWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        top, left, width, height = 400,400,800,600
        self.setGeometry(top, left, width, height)
        self.setWindowTitle("Relevance Painter")
        self.setCentralWidget(RelevancePainter((top,left,width,height)))
        self.setFixedSize(width,height)


class RelevancePainter(QWidget):
    def __init__(self,geometry):
        super().__init__()
        # set layout of painter

        layout = QGridLayout()
        self.spacing = 10
        layout.setSpacing(self.spacing)
        # canvas widget that displays image and facilitates drawing
        self.canvas = RelevanceCanvas("me_and_gimli.jpg",geometry+(self.spacing,))
        # button widget for reading in video dialogue
        loadFileBtn = QPushButton("Read in video", self)
        loadFileBtn.clicked.connect(self.loadFile)

        layout.addWidget(self.canvas,1,0,3,3)
        layout.addWidget(loadFileBtn, 4,3,1,1)
        self.setLayout(layout)

    def loadFile(self, path):
        name = QFileDialog.getOpenFileName(self, "Open Video")[0]
        self.canvas.setImage(name)



class RelevanceCanvas(QWidget):
    def __init__(self, img,mainWindow):
        super().__init__()

        # set the background to the initial frame
        self.image = QImage(img)

        # "colour modes" for frame, i.e. relevant or irrelevant
        self.falseColor = QColor(0,0,0,127)
        self.trueColor = Qt.transparent

        # set opaque overlay for frame
        self.drawOverlay = QImage(self.size(),QImage.Format_ARGB32)
        self.drawOverlay.fill(self.falseColor)

        self.drawing = False
        self.cursor = True

        # default brushes
        self._clear_size = 60

        self.setCursor()

        # set keybindings for reveal and replace modes
        self.shortcut = QShortcut(QKeySequence("E"), self)
        self.shortcut.activated.connect(self.setDrawCursor)
        self.shortcut = QShortcut(QKeySequence("D"), self)
        self.shortcut.activated.connect(self.setEraseCursor)


    def resizeEvent(self,event):
        # We need to update the widgets geometry to account for it being nested in a main window
        overlayR = QRect(0,0,self.drawOverlay.width(),self.drawOverlay.height())
        r = self.geometry()
        # self.layoutRatio = (overlayR.width()/parR.width(),overlayR.height()/parR.height())
        self.layoutRatio = (overlayR.width()/r.width(),overlayR.height()/r.height())
        # store offsets, and ratios for transforming mouse pos
        super().resizeEvent(event)


    def enterEvent(self,event):
        self.setCursor()
        super().enterEvent(event)
    def leaveEvent(self,event):
        self.setCursor()
        super().leaveEvent(event)

    def paintEvent(self,event):
        # draw frame and overlay
        canvasPainter = QPainter(self)
        canvasPainter.drawImage(self.rect(),self.image,self.image.rect())
        canvasPainter.drawImage(self.rect(),self.drawOverlay,self.drawOverlay.rect())

    def mousePressEvent(self,event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.lastPoint = event.pos()

    def mouseMoveEvent(self,event):
        if event.buttons() and Qt.LeftButton and self.drawing:
            # painter.QPen(QColor(0,0,0,127))
            # rect = QRect(QPoint(), self._clear_size*QSize())
            rect = QRect(
                QPoint(),
                QSize(
                    int(self._clear_size)+5,
                    int(self._clear_size)-6,
                )
            )
            x, y = event.pos().x()*self.layoutRatio[0], event.pos().y()*self.layoutRatio[1]
            epos = QPoint(x,y)
            # epos = event.pos()

            rect.moveCenter(epos)
            bg = self.drawOverlay
            painter = QPainter(bg)
            # painter.translate(self.left,self.top)
            # painter.scale(self.height,self.width)
            if self.cursor:
                for y in range(rect.top(),rect.bottom()):
                    for x in range(rect.left(),rect.right()):
                        pos = QPoint(x,y)
                        if not bg.valid(pos):
                            continue
                        colour_at_pos = self.falseColor if bg.pixelColor(pos) == self.trueColor else self.trueColor
                        # print(colour_at_pos)
                        painter.setPen(QPen(colour_at_pos))
                        painter.drawPoint(pos)
                # painter.drawRect(rect)
            else:
                painter.save()
                painter.setCompositionMode(QPainter.CompositionMode_Clear)
                painter.eraseRect(rect)
                # painter.fillRect(rect,QColor(0,0,0,127))
                painter.restore()
            self.lastPoint = epos
            self.update()
            painter.end()


    def mouseReleaseEvent(self,event):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def setDrawCursor(self):
        self.cursor = True
        self.setCursor()
    def setEraseCursor(self):
        self.cursor = False
        self.setCursor()

    def setCursor(self):
        if self.underMouse():
            pixmap = QPixmap(QSize(1,1)*self._clear_size)
            if self.cursor:
                pixmap.fill(self.falseColor)
            else:
                pixmap.fill(Qt.transparent)
            painter = QPainter(pixmap)
            painter.setPen(QPen(Qt.black,2))
            painter.drawRect(pixmap.rect())
            painter.end()
            cursor = QCursor(pixmap)
            QApplication.setOverrideCursor(cursor)
        else:
            while QApplication.overrideCursor():
                QApplication.restoreOverrideCursor()

    def setImage(self,path):
        _, frames = video.get_input(path)
        print(len(frames))
        key_frames = []
        skip = len(frames[0])//4
        for batch in frames:
            for i in range(0,len(batch),skip):
                key_frames.append(batch[i])
        self.frames = key_frames
        cvImg = cv2.cvtColor(key_frames[10],cv2.COLOR_BGR2RGB)
        height, width, channel = cvImg.shape
        bytesPerLine = 3 * width
        self.image = QImage(cvImg.data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.update()


def set_dark_fusion(qApp):
    #https://gist.github.com/lschmierer/443b8e21ad93e2a2d7eb#file-dark_fusion-py
    qApp.setStyle("Fusion")

    dark_palette = QPalette()

    dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.WindowText, Qt.white)
    dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
    dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
    dark_palette.setColor(QPalette.ToolTipText, Qt.white)
    dark_palette.setColor(QPalette.Text, Qt.white)
    dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ButtonText, Qt.white)
    dark_palette.setColor(QPalette.BrightText, Qt.red)
    dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.HighlightedText, Qt.black)

    qApp.setPalette(dark_palette)

    qApp.setStyleSheet("QToolTip { color: #ffffff; background-color: #2a82da; border: 1px solid white; }")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    set_dark_fusion(app)

    window = AppWindow()
    window.show()

    app.exec_()
