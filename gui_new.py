#!/usr/bin/env python3

import sys
from PyQt5.QtWidgets import QWidget, QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QGridLayout, QShortcut
from PyQt5.QtGui import QPixmap, QImage, QColor, QPainter, QPen, QCursor, QKeySequence, QPalette
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

        # set keybindings for reveal and replace modes
        self.shortcut = QShortcut(QKeySequence("D"), self)
        self.shortcut.activated.connect(self.setDrawCursor)
        self.shortcut = QShortcut(QKeySequence("E"), self)
        self.shortcut.activated.connect(self.setEraseCursor)

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
            rect = QRect(QPoint(), self._clear_size*QSize())
            rect.moveCenter(event.pos())
            bg = self.drawOverlay
            painter = QPainter(bg)
            if self.cursor:
                # get rectangle positions
                for y in range(rect.bottom(),rect.top()):
                    for x in range(rect.right(),rect.left()):
                        pos = QPoint(x,y)
                        colour_at_pos = self.falseColor if bg.pixelColor(x,y) == self.trueColor else self.trueColor
                        # print(colour_at_pos)
                        painter.setPen(QPen(colour_at_pos))
                        painter.drawPoint(x,y)
                # painter.drawRect(rect)
            else:
                painter.save()
                painter.setCompositionMode(QPainter.CompositionMode_Clear)
                painter.eraseRect(rect)
                # painter.fillRect(rect,QColor(0,0,0,127))
                painter.restore()
            self.lastPoint = self.mapToGlobal(event.pos())
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
