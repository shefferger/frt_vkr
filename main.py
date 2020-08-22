import time
import cv2
import os
import pickle
import shutil
import numpy as np
from datetime import datetime
from PIL import Image
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox, QLabel, QMainWindow, QPushButton, QComboBox

pathFolder = "users/"
stopSig = False
font_ = cv2.FONT_HERSHEY_COMPLEX
currentUsers = {}
usermap = {}
reportFile = ""
fps = 30
width = 1280
height = 720

recognizer = cv2.face_FisherFaceRecognizer.create()
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


if not os.path.exists(pathFolder):
    os.makedirs(pathFolder)

if not os.path.exists("reports/"):
    os.makedirs("reports/")

with open("usermap.pkl", "a"):
    os.utime("usermap.pkl", None)


def getCsvNum(a=1):
    _reportFile = "reports/report " + str(datetime.now().strftime('%Y-%m-%d   %H-%M')) + " (" + str(a) + ")" + ".csv"
    if os.path.exists(_reportFile):
        _reportFile = getCsvNum(a + 1)
    return _reportFile


def initCsv():
    global reportFile
    reportFile = getCsvNum()
    with open(reportFile, "a") as f:
        os.utime(reportFile, None)
        f.write('Время;' + 'ID;' + 'Группа;' + 'Имя;' + 'Фамилия;\n')


def save(data):
    with open("usermap.pkl", "wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def load():
    try:
        with open("usermap.pkl", "rb") as f:
            return pickle.load(f)
    except EOFError:
        dummy = {0: ['Имя', 'Фамилия', 'Группа']}
        save(dummy)
        return dummy


def getImgNum(_id, _num):
    if os.path.exists(pathFolder + "id_" + str(_id) + "/" + str(_id) + "." + str(_num) + ".jpg"):
        _num += 1
        _num = getImgNum(_id, _num)
    return _num


def capture(name='', surname='', group='', x='0'):
    global usermap
    usermap = load()
    count = 0
    try:
        x = int(x)
        if x != 0:
            lastid = x
            if x in usermap:
                userData = usermap.get(lastid)
            else:
                return
        else:
            userData = [name, surname, group]
            lastid = list(usermap.keys())[-1] + 1
            usermap.update({lastid: userData})
            save(usermap)
    except Exception:
        return

    cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
    cap.set(3, width)
    cap.set(4, height)
    cap.set(5, fps)
    time.sleep(1)
    if not cap.isOpened():
        return
    while True:
        success, img = cap.read()
        if not success:
            return
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Capturing", gray_img)
        faces = faceCascade.detectMultiScale(gray_img, 1.3, 14)
        for (x, y, w, h) in faces:
            path = "users/id_" + str(lastid)
            if not os.path.exists(path):
                os.makedirs(path)
            cv2.rectangle(gray_img, (x, y), (x + w, y + h), 255, 1)
            cv2.putText(gray_img,
                        "ID: " + str(lastid) + ", Имя: " + userData[0] + ", обработано: " + str(count * 5 / 6) + "%",
                        (x, y + h + 22), font_, 0.5, 255, 1, cv2.LINE_AA)
            if count >= 120:
                cap.release()
                cv2.destroyWindow('Capturing')
                return True, "Лицо пользователя \'" + userData[0] + "\' успешно записано!"
            count += 1
            num = getImgNum(lastid, count)
            totalImg = gray_img[y:y + h, x:x + w]
            if not (w == 128 & h == 128):
                totalImg = cv2.resize(totalImg, (128, 128))
            cv2.imwrite("users/" + "id_" + str(lastid) + '/' + str(lastid) + '.' + str(num) + ".jpg", totalImg)
        cv2.imshow('Capturing', gray_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyWindow('Capturing')
            return False, "Захват прерван"


def processRecon(widg, isToBeWrite, isToBeShown):
    global usermap, stopSig, currentUsers, reportFile, width, height, fps
    if not os.path.exists("trainer/trainer.yml"):
        showDialog("Ошибка", "Сначала необходимо обучить модель!")
        return False
    recognizer.read('trainer/trainer.yml')
    usermap = load()
    if isToBeWrite:
        initCsv()
    cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
    cap.set(3, width)
    cap.set(4, height)
    cap.set(5, fps)
    time.sleep(1)
    while True:
        ctime = datetime.now()
        ret, im = cap.read()
        gray_img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray_img, 1.3, 14)
        for (x, y, w, h) in faces:
            face_id = recognizer.predict(cv2.resize(gray_img[y + 2:y + h + 2, x - 2:x + w - 2], (128, 128)))
            if face_id[0] in usermap:
                userData = usermap.get(face_id[0])
            else:
                userData = ['n/a', 'n/a', 'n/a']
            conf = face_id[1]
            cv2.rectangle(gray_img, (x, y), (x + w, y + h), 240, 1)
            if conf >= 500:
                infoText = "(Возможна ошибка)"
            else:
                if face_id[0] not in currentUsers:
                    currentUsers.update({face_id[0]: 0})
                else:
                    if currentUsers.get(face_id[0]) < 6:
                        currentUsers.update({face_id[0]: currentUsers.get(face_id[0]) + 1})
                    else:
                        if currentUsers.get(face_id[0]) != 100:
                            widg.addItem(ctime.strftime('%d-%m-%Y  %H:%M:%S') + "\n" + userData[2] +
                                         "\n" + userData[0] + "  " + userData[1] + "\n")
                            currentUsers.update({face_id[0]: 100})
                            if isToBeWrite:
                                with open(reportFile, 'a') as f:
                                    f.write(ctime.strftime('%d-%m-%Y  %H:%M:%S') + ";" + str(face_id[0]) + ";" +
                                            userData[2] + ";" + userData[0] + ";" + userData[1] + "\n")
                infoText = ""
            if isToBeShown:
                cv2.putText(gray_img, "Имя: " + userData[0] + ", Фамилия: " + userData[1], (x, y + h + 24), font_, 0.5,
                            255,
                            1, cv2.LINE_AA)
                cv2.putText(gray_img, "Группа: " + userData[2], (x, y + h + 40), font_, 0.5, 255, 1, cv2.LINE_AA)
                cv2.putText(gray_img, "conf=" + str(round(conf, 2)) + ", id: " + str(face_id[0]), (x, y + h + 56),
                            font_,
                            0.5, 255, 1, cv2.LINE_AA)
                cv2.putText(gray_img, infoText, (x, y + h + 72), font_, 0.5, 255, 1, cv2.LINE_AA)
        if isToBeShown:
            cv2.imshow('Recognition', gray_img)
        if (cv2.waitKey(1) & 0xFF == ord('q')) or stopSig:
            stopSig = False
            cap.release()
            cv2.destroyWindow('Recognition')
            return True


def getImagesAndLabels():
    imgDirs = next(os.walk(pathFolder + "."))[1]
    imagePaths = []
    faceSamples = []
    ids = []

    for i in imgDirs:
        for j in next(os.walk(pathFolder + i + "/."))[2]:
            imagePaths.append(pathFolder + i + "/" + j)

    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img, 'uint8')
        id_ = int(os.path.split(imagePath)[-1].split(".")[0])
        faceSamples.append(img_numpy)
        ids.append(id_)
    return faceSamples, ids


def train():
    global usermap
    usermap = load()
    if len(usermap) <= 2:
        return False
    if not os.path.exists("trainer/"):
        os.makedirs("trainer/")
    try:
        faces, ids = getImagesAndLabels()
        recognizer.train(faces, np.array(ids))
        recognizer.save('trainer/trainer.yml')
    except Exception:
        return False
    return True


def processReconStop():
    global stopSig
    stopSig = True


def resetJournal(widg, isToBeWrite):
    global currentUsers
    currentUsers = {}
    widg.clear()
    if isToBeWrite:
        initCsv()


def showDialog(title, text):
    msgBox = QMessageBox()
    msgBox.setIcon(QMessageBox.Information)
    msgBox.setText(text)
    msgBox.setWindowTitle(title)
    msgBox.setStandardButtons(QMessageBox.Ok)
    msgBox.exec()


class SecondWindow(QMainWindow):
    def __init__(self):
        super(SecondWindow, self).__init__()
        global usermap
        usermap = load()

        self.resize(380, 380)
        self.setWindowFlag(QtCore.Qt.WindowCloseButtonHint, False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        self.setSizePolicy(sizePolicy)
        self.setMinimumSize(QtCore.QSize(380, 380))
        self.setMaximumSize(QtCore.QSize(380, 380))

        self.lb_1 = QLabel('Выберите пользователя:', self)
        self.lb_1.setGeometry(QtCore.QRect(11, 10, 360, 15))

        self.btn_1 = QPushButton('Сохранить', self)
        self.btn_1.setGeometry(QtCore.QRect(11, 250, 175, 35))
        self.btn_1.clicked.connect(self.saveUData)

        self.btn_2 = QPushButton('Удалить', self)
        self.btn_2.setGeometry(QtCore.QRect(11, 290, 175, 35))
        self.btn_2.clicked.connect(self.deleteUser)

        self.btn_3 = QPushButton('Добавить фото\n(с камеры)', self)
        self.btn_3.setGeometry(QtCore.QRect(196, 290, 175, 35))
        self.btn_3.clicked.connect(self.capPrepair)

        self.btn_4 = QPushButton('< Назад', self)
        self.btn_4.setGeometry(QtCore.QRect(11, 340, 360, 30))
        self.btn_4.clicked.connect(self.closeSelf)

        self.lineEdit = QtWidgets.QLineEdit(self)
        self.lineEdit.setGeometry(QtCore.QRect(11, 90, 165, 24))
        self.lineEdit.setEnabled(False)

        self.lineEdit_2 = QtWidgets.QLineEdit(self)
        self.lineEdit_2.setGeometry(QtCore.QRect(11, 150, 165, 24))

        self.lineEdit_3 = QtWidgets.QLineEdit(self)
        self.lineEdit_3.setGeometry(QtCore.QRect(206, 90, 165, 24))

        self.lineEdit_4 = QtWidgets.QLineEdit(self)
        self.lineEdit_4.setGeometry(QtCore.QRect(206, 150, 165, 24))

        self.cmb_1 = QComboBox(self)
        self.cmb_1.setGeometry((QtCore.QRect(11, 30, 360, 22)))
        self.cmb_1.currentTextChanged.connect(self.comboSelected)

        for i in usermap.keys():
            self.cmb_1.addItem(str(i) + ": " + usermap.get(i)[2]
                               + " -   " + usermap.get(i)[0] + " " + usermap.get(i)[1])
        self.cmb_1.removeItem(0)

        self.lb_2 = QLabel('ID', self)
        self.lb_2.setGeometry(QtCore.QRect(12, 74, 180, 15))

        self.lb_3 = QLabel('Группа', self)
        self.lb_3.setGeometry(QtCore.QRect(207, 74, 180, 15))

        self.lb_4 = QLabel('Имя', self)
        self.lb_4.setGeometry(QtCore.QRect(12, 134, 180, 15))

        self.lb_5 = QLabel('Фамилия', self)
        self.lb_5.setGeometry(QtCore.QRect(207, 134, 180, 15))

    def comboSelected(self, val):
        uid = int(val.split(':')[0])
        self.lineEdit.setText(str(uid))
        self.lineEdit_2.setText(usermap.get(uid)[0])
        self.lineEdit_3.setText(usermap.get(uid)[2])
        self.lineEdit_4.setText(usermap.get(uid)[1])

    def closeSelf(self):
        self.close()
        MainWindow.show()

    def capPrepair(self):
        uidStr = self.lineEdit.text()
        capture(x=uidStr)

    def saveUData(self):
        uid = int(self.lineEdit.text())
        usermap.update({uid: [self.lineEdit_2.text(),
                              self.lineEdit_4.text(),
                              self.lineEdit_3.text()]})
        save(usermap)
        self.cmb_1.setItemText(self.cmb_1.currentIndex(), self.lineEdit.text() + ": " + usermap.get(uid)[2]
                               + " -   " + usermap.get(uid)[0] + " " + usermap.get(uid)[1])

    def deleteUser(self):
        global usermap
        uid = int(self.lineEdit.text())
        del usermap[uid]
        save(usermap)
        self.cmb_1.removeItem(self.cmb_1.currentIndex())
        shutil.rmtree("users/id_" + str(uid), ignore_errors=True)


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(380, 445)

        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMinimumSize(QtCore.QSize(380, 445))
        MainWindow.setMaximumSize(QtCore.QSize(380, 445))
        font = QtGui.QFont()
        font.setPointSize(9)
        MainWindow.setFont(font)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(0, 0, 380, 400))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.tabWidget.setFont(font)
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.checkBox = QtWidgets.QCheckBox(self.tab)
        self.checkBox.setGeometry(QtCore.QRect(10, 10, 240, 16))
        self.checkBox.setChecked(True)
        self.checkBox.setObjectName("checkBox")
        self.pushButton_4 = QtWidgets.QPushButton(self.tab)
        self.pushButton_4.setGeometry(QtCore.QRect(210, 10, 160, 40))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.pushButton_4.setFont(font)
        self.pushButton_4.setObjectName("pushButton_4")
        self.listWidget = QtWidgets.QListWidget(self.tab)
        self.listWidget.setGeometry(QtCore.QRect(10, 60, 360, 270))
        self.listWidget.viewport().setProperty("cursor", QtGui.QCursor(QtCore.Qt.IBeamCursor))
        self.listWidget.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.listWidget.setFrameShadow(QtWidgets.QFrame.Plain)
        self.listWidget.setMidLineWidth(1)
        self.listWidget.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.listWidget.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.listWidget.setLayoutMode(QtWidgets.QListView.Batched)
        self.listWidget.setItemAlignment(QtCore.Qt.AlignTop)
        self.listWidget.setObjectName("listWidget")
        self.pushButton_5 = QtWidgets.QPushButton(self.tab)
        self.pushButton_5.setGeometry(QtCore.QRect(10, 340, 360, 30))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.pushButton_5.setFont(font)
        self.pushButton_5.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.pushButton_5.setObjectName("pushButton_5")

        self.pushButton_6 = QtWidgets.QPushButton(MainWindow)
        self.pushButton_6.setGeometry(QtCore.QRect(11, 405, 360, 30))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.pushButton_6.setFont(font)
        self.pushButton_6.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.pushButton_6.setObjectName("pushButton_6")

        self.checkBox_3 = QtWidgets.QCheckBox(self.tab)
        self.checkBox_3.setGeometry(QtCore.QRect(10, 30, 191, 17))
        self.checkBox_3.setChecked(True)
        self.checkBox_3.setObjectName("checkBox_3")
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.pushButton_2 = QtWidgets.QPushButton(self.tab_2)
        self.pushButton_2.setGeometry(QtCore.QRect(100, 230, 180, 40))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton = QtWidgets.QPushButton(self.tab_2)
        self.pushButton.setGeometry(QtCore.QRect(100, 130, 180, 40))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.line = QtWidgets.QFrame(self.tab_2)
        self.line.setGeometry(QtCore.QRect(10, 170, 350, 20))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.lineEdit = QtWidgets.QLineEdit(self.tab_2)
        self.lineEdit.setGeometry(QtCore.QRect(100, 10, 180, 20))
        self.lineEdit.setObjectName("lineEdit")
        self.lineEdit_2 = QtWidgets.QLineEdit(self.tab_2)
        self.lineEdit_2.setGeometry(QtCore.QRect(100, 40, 180, 20))
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.lineEdit_3 = QtWidgets.QLineEdit(self.tab_2)
        self.lineEdit_3.setGeometry(QtCore.QRect(100, 70, 180, 20))
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.label = QtWidgets.QLabel(self.tab_2)
        self.label.setGeometry(QtCore.QRect(60, 10, 50, 15))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label.setFont(font)
        self.label.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.tab_2)
        self.label_2.setGeometry(QtCore.QRect(30, 40, 60, 16))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.tab_2)
        self.label_3.setGeometry(QtCore.QRect(40, 70, 50, 16))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.line_2 = QtWidgets.QFrame(self.tab_2)
        self.line_2.setGeometry(QtCore.QRect(10, 210, 350, 16))
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.labelCapture = QtWidgets.QLabel(self.tab_2)
        self.labelCapture.setGeometry(QtCore.QRect(10, 190, 350, 16))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.labelCapture.setFont(font)
        self.labelCapture.setText("")
        self.labelCapture.setObjectName("labelCapture")
        self.checkBox_2 = QtWidgets.QCheckBox(self.tab_2)
        self.checkBox_2.setGeometry(QtCore.QRect(100, 100, 230, 17))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.checkBox_2.setFont(font)
        self.checkBox_2.setChecked(True)
        self.checkBox_2.setObjectName("checkBox_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.tab_2)
        self.pushButton_3.setGeometry(QtCore.QRect(100, 320, 180, 41))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.pushButton_3.setFont(font)
        self.pushButton_3.setObjectName("pushButton_3")
        self.tabWidget.addTab(self.tab_2, "")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.tabWidget.addTab(self.tab_3, "")

        self.label_4 = QtWidgets.QLabel(self.tab_3)
        self.label_4.setGeometry(QtCore.QRect(11, 10, 120, 15))
        self.label_4.setText("Разрешение камеры:")

        self.label_5 = QtWidgets.QLabel(self.tab_3)
        self.label_5.setGeometry(QtCore.QRect(11, 60, 120, 15))
        self.label_5.setText("Фреймрейт камеры:")

        self.lineEdit_4 = QtWidgets.QLineEdit(self.tab_3)
        self.lineEdit_4.setGeometry(QtCore.QRect(121, 10, 60, 20))
        self.lineEdit_4.setText("1280")

        self.lineEdit_5 = QtWidgets.QLineEdit(self.tab_3)
        self.lineEdit_5.setGeometry(QtCore.QRect(121, 35, 60, 20))
        self.lineEdit_5.setText("720")

        self.lineEdit_6 = QtWidgets.QLineEdit(self.tab_3)
        self.lineEdit_6.setGeometry(QtCore.QRect(121, 60, 60, 20))
        self.lineEdit_6.setText("30")

        self.pushButton_7 = QtWidgets.QPushButton(self.tab_3)
        self.pushButton_7.setGeometry(QtCore.QRect(81, 90, 140, 25))
        self.pushButton_7.setText("Сохранить настройки")

        MainWindow.setCentralWidget(self.centralwidget)
        MainWindow.setWindowFlag(QtCore.Qt.WindowCloseButtonHint, False)
        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.configureUI()

    def paramSaveBtn(self):
        global width, height, fps
        try:
            width, height = int(self.lineEdit_4.text()), int(self.lineEdit_5.text())
            fps = int(self.lineEdit_6.text())
        except TypeError:
            width, height, fps = 640, 480, 30
            showDialog("Ошибка ввода!", "Ошибка ввода!")

    def button_pressed(self):
        if (self.lineEdit.text() == "") and (self.lineEdit_2.text() == ""):
            showDialog("Ошибка", "Необходимо ввести хотя-бы имя или фамилию!")
            return
        self.tab.setEnabled(False)
        self.tab_2.setEnabled(False)
        isOk, infoText = capture(self.lineEdit.text(), self.lineEdit_2.text(), self.lineEdit_3.text(), 0)
        self.labelCapture.setText(infoText)
        self.lineEdit.clear()
        self.lineEdit_2.clear()
        self.lineEdit_3.clear()
        self.tab.setEnabled(True)
        self.tab_2.setEnabled(True)

    def button2_pressed(self):
        self.labelCapture.setText("Идет обучение...")
        self.tab.setEnabled(False)
        self.tab_2.setEnabled(False)
        self.tabWidget.repaint()
        res = train()
        self.tab.setEnabled(True)
        self.tab_2.setEnabled(True)
        if not res:
            self.labelCapture.setText("Обучение завершено с ошибкой")
            showDialog("Ошибка", "Для обучения необходимо 2 и более записанных лица")
        else:
            self.labelCapture.setText("Обучение завершено!")

    def button3_pressed(self):
        global usermap
        usermap = load()
        if len(usermap) <= 2:
            showDialog("Ошибка", "Нет записанных лиц!\nСначала запишите минимум 2 лица")
            return
        self.faceTool = SecondWindow()
        self.faceTool.show()
        MainWindow.hide()

    def button4_pressed(self):
        self.togTabs()
        st = True
        if self.pushButton_4.text() == "Закончить распознавание":
            self.pushButton_4.setText("Начать распознавание")
            processReconStop()
        else:
            self.pushButton_4.setText("Закончить распознавание")
            st = processRecon(self.listWidget, self.checkBox.isChecked(), self.checkBox_3.isChecked())
        if not st:
            self.pushButton_4.setText("Начать распознавание")
            self.togTabs()

    def togTabs(self):
        self.checkBox.setEnabled(not self.checkBox.isEnabled())
        self.checkBox_3.setEnabled(not self.checkBox_3.isEnabled())
        self.tab_2.setEnabled(not self.tab_2.isEnabled())

    def button5_pressed(self):
        resetJournal(self.listWidget, self.checkBox.isChecked())

    def button6_pressed(self):
        global stopSig
        stopSig = True
        QtWidgets.qApp.quit()

    def configureUI(self):
        self.pushButton.clicked.connect(self.button_pressed)
        self.pushButton_2.clicked.connect(self.button2_pressed)
        self.pushButton_3.clicked.connect(self.button3_pressed)
        self.pushButton_4.clicked.connect(self.button4_pressed)
        self.pushButton_5.clicked.connect(self.button5_pressed)
        self.pushButton_6.clicked.connect(self.button6_pressed)
        self.pushButton_7.clicked.connect(self.paramSaveBtn)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Система контроля посещаемости"))
        self.checkBox.setText(_translate("MainWindow", "Вести учет в документ"))
        self.pushButton_4.setText(_translate("MainWindow", "Начать распознавание"))
        self.pushButton_5.setText(_translate("MainWindow", "Сбросить журнал и начать новый файл"))
        self.pushButton_6.setText(_translate("MainWindow", "Выход"))
        self.checkBox_3.setText(_translate("MainWindow", "Показывать камеру"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Распознавание лиц"))
        self.pushButton_2.setText(_translate("MainWindow", "Обучить нейросеть"))
        self.pushButton.setText(_translate("MainWindow", "Записать образец\nлица"))
        self.label.setText(_translate("MainWindow", "Имя:"))
        self.label_2.setText(_translate("MainWindow", "Фамилия:"))
        self.label_3.setText(_translate("MainWindow", "Группа:"))
        self.checkBox_2.setText(_translate("MainWindow", "Показывать камеру"))
        self.pushButton_3.setText(_translate("MainWindow", "Обновить информацию\n"
                                                           "о пользователе"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Обновление базы лиц"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("MainWindow", "Настройки"))


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)

    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
