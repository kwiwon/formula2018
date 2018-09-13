import numpy
import os
import datetime
import cv2
from PIL import Image
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation, Softmax
from keras.utils import np_utils
from matplotlib import pyplot as plt


DETECT_INTER = 5
IMAGE_SIZE = 80
model_path = os.path.join(os.getcwd(), 'traffic_sign', 'model5_cut32_20.h5')
classes = {'None': 0,
           'ForkLeft': 1,
           'ForkRight': 2,
           'LaneLeft': 3,
           'LaneRight': 4,
           'TurnLeft': 5,
           'TurnRight': 6,
           'ULeft': 7,
           'URight': 8}
inv_classes = {v: k for k, v in classes.items()}


class identifyTrafficSign(object):
    def __init__(self):
        self.model = self.loadModel()
        self.count = DETECT_INTER

    def detect(self, img):
        self.count -= 1
        if self.count == 0:
            self.count = DETECT_INTER
            imgCut, bound = processImg(img, "detect")
            img_2D = imgCut.reshape(1, imgCut.shape[0], imgCut.shape[1], 1).astype('float32')
            img_norm = img_2D / 255
            predictions = self.model.predict_classes(img_norm)
            # if classes['None'] not in list(predictions):
            #     self.createOverlap(img, predictions, bound)
            #     self.savePic(img)
            return inv_classes[predictions[0]]
        return None

    def createOverlap(self, img, predict, bound):
        plt.clf()
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        curAxis = plt.gca()

        curAxis.axis('off')
        x, y, w, h = bound
        coords = (x, y), w, h
        curAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor='red', linewidth=3))
        curAxis.text(0, 0, inv_classes[predict[0]], bbox={'facecolor': 'red', 'alpha': 1})
        plt.show()

    def savePic(self, img):
        saveDir = "../Data/rowData/"
        now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
        filename = "row_{}.jpg".format(now)
        savefile = Image.fromarray(numpy.uint8(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        savefile.save(os.path.join(saveDir, filename))

    @staticmethod
    def loadModel(modelPath=model_path):
        return load_model(modelPath)

class trainTrafficSign(object):
    train_path = "/Users/mavis/Documents/projects/AIContest/Formula/Data/TrainData/"
    valid_path = "/Users/mavis/Documents/projects/AIContest/Formula/Data/VerifyData/"

    def __init__(self):
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None
        self.model = None

    def loadData(self):
        for dir, dirnames, fileNames in os.walk(self.train_path):
            for dirname in dirnames:
                dirpath = os.path.join(dir, dirname)
                for subdir, subdirnames, subfileNames in os.walk(dirpath):
                    for f in subfileNames:
                        if '.jpg' not in f:
                            continue
                        filePath = os.path.join(subdir, f)
                        jpgfile = Image.open(filePath)
                        imgOrg = numpy.asarray(jpgfile)
                        imgAft, _ = processImg(imgOrg)
                        if self.X_train is None:
                            self.X_train = numpy.array([imgAft])
                            self.Y_train = numpy.array([classes[dirname]])
                        else:
                            self.X_train = numpy.vstack((self.X_train, numpy.array([imgAft])))
                            self.Y_train = numpy.append(self.Y_train, classes[dirname])
        randomize = numpy.arange(len(self.X_train))
        numpy.random.shuffle(randomize)
        self.X_train = self.X_train[randomize]
        self.Y_train = self.Y_train[randomize]
        print("Train Data (X_train image): {}".format(self.X_train.shape))
        print("Train Data (Y_train label): {}".format(self.Y_train.shape))

        for dir, dirnames, fileNames in os.walk(self.valid_path):
            for dirname in dirnames:
                dirpath = os.path.join(dir, dirname)
                for subdir, subdirnames, subfileNames in os.walk(dirpath):
                    for f in subfileNames:
                        if '.jpg' not in f:
                            continue
                        filePath = os.path.join(subdir, f)
                        jpgfile = Image.open(filePath)
                        imgOrg = numpy.asarray(jpgfile)
                        imgAft, _ = processImg(imgOrg)
                        if self.X_test is None:
                            self.X_test = numpy.array([imgAft])
                            self.Y_test = numpy.array([classes[dirname]])
                        else:
                            self.X_test = numpy.vstack((self.X_test, numpy.array([imgAft])))
                            self.Y_test = numpy.append(self.Y_test, classes[dirname])
        print("Train Data (X_test image): {}".format(self.X_test.shape))
        print("Train Data (Y_test label): {}".format(self.Y_test.shape))

    def train(self):
        self.model = Sequential()
        shape = self.X_train.shape[1], self.X_train.shape[2]
        self.model.add(Conv2D(filters=16, kernel_size=(5,5), padding='same',
                              input_shape=(shape[0], shape[1], 1), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))

        self.model.add(Conv2D(filters=36, kernel_size=(5,5), padding='same',activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))

        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(9, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


        y_TrainOneHot = np_utils.to_categorical(self.Y_train)
        y_TestOneHot = np_utils.to_categorical(self.Y_test)

        X_train_2D = self.X_train.reshape(self.X_train.shape[0], shape[0], shape[1], 1).astype('float32')
        X_test_2D = self.X_test.reshape(self.X_test.shape[0], shape[0], shape[1], 1).astype('float32')

        x_Train_norm = X_train_2D / 255
        x_Test_norm = X_test_2D / 255

        train_history = self.model.fit(x=x_Train_norm, y=y_TrainOneHot, validation_split=0.2, epochs=20,
                                       batch_size=10, verbose=2)

        scores = self.model.evaluate(x_Test_norm, y_TestOneHot)
        print()
        print("\t[Info] Accuracy of testing data = {:2.1f}%".format(scores[1] * 100.0))

        predictions = self.model.predict_classes(x_Test_norm, verbose=1)
        # get prediction result
        print(predictions)

        """for x in range(self.X_test.shape[0]):
            plt.imshow(self.X_test[x])
            plt.show()"""

        self.model.summary()

    def save(self):
        self.model.save(model_path)

def processImg(srcImg, srcType="train"):
    img = srcImg[0:IMAGE_SIZE]
    imgGray = procColor(img, srcType)
    imgCut2, bounds = cropImg2(imgGray)
    return imgCut2, bounds

def procColor(srcImg, srcType):
    # mask all colors except red (traffic sign is in red)
    if srcType == "train": # src img is rgb
        maskRed = cv2.inRange(srcImg, (70, -1, -1), (256, 60, 60))
    else: # type is detect, src img is bgr
        maskRed = cv2.inRange(srcImg, (-1, -1, 70), (60, 60, 256))
    target = cv2.bitwise_and(srcImg, srcImg, mask=maskRed)
    ret, imgGray = cv2.threshold(cv2.cvtColor(target, cv2.COLOR_RGB2GRAY), 30, 255, cv2.THRESH_BINARY)
    return imgGray

def cropImg(orgImg):
    CUT_SIZE = (40, 40)
    cols = numpy.sum(orgImg/255, axis=0)
    rows = numpy.sum(orgImg/255, axis=1)

    grater_mean_col = numpy.where(cols > cols.mean())[0]
    if grater_mean_col.size > orgImg.shape[0] / 2:
        grater_mean_col = numpy.where(cols > cols.mean() + cols.std())[0]
    grater_mean_row = numpy.where(rows > rows.mean())[0]
    if grater_mean_row.size > orgImg.shape[1] / 2:
        grater_mean_row = numpy.where(rows > rows.mean() + rows.std())[0]

    tar_col = consecutiveMax(grater_mean_col, orgImg.shape[0]/10)
    tar_row = consecutiveMax(grater_mean_row, orgImg.shape[1]/10)
    if tar_col is None or tar_row is None:
        cutImg = orgImg[0:CUT_SIZE[0], 0:CUT_SIZE[1]]
    else:
        cutImg_col_len = tar_col[-1] - tar_col[0]
        cutImg_col_mean = int((tar_col[-1] + tar_col[0])/2)
        cutImg_row_len = tar_row[-1] - tar_row[0]
        cutImg_row_mean = int((tar_row[-1] + tar_row[0])/2)
        cut_size = cutImg_col_len if cutImg_col_len > cutImg_row_len else cutImg_row_len
        cut_size = int(cut_size / 2) + 5
        cutImg_col_mean = cut_size if cutImg_col_mean < cut_size else cutImg_col_mean
        cutImg_col_mean = orgImg.shape[1]-cut_size if cutImg_col_mean > orgImg.shape[1]-cut_size else cutImg_col_mean
        cutImg_row_mean = cut_size if cutImg_row_mean < cut_size else cutImg_row_mean
        cutImg_row_mean = orgImg.shape[0]-cut_size if cutImg_row_mean > orgImg.shape[0]-cut_size else cutImg_row_mean

        cutImg = orgImg[cutImg_row_mean - cut_size:cutImg_row_mean + cut_size,
                 cutImg_col_mean - cut_size:cutImg_col_mean + cut_size]

    return cv2.resize(cutImg, CUT_SIZE, 0, 0)

def cropImg2(greyImg):
    # find contours first
    image, contours, hier = cv2.findContours(greyImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # find bounding box of traffic signs
    bounding_rects = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        bounding_rects.append([x, y, w, h])

    bounding_rects = sorted(bounding_rects, key=lambda x: x[0])

    # merge bounding boxes (one traffic sign could be separated into two boxes, need to merge)
    merged_bounding_boxes = []
    final_bounding_boxes = []
    for b in range(len(bounding_rects)):
        # get the bounding rect
        x, y, w, h = bounding_rects[b]
        if w > 5 and h > 5:
            if len(merged_bounding_boxes) > 0:
                if x > merged_bounding_boxes[-1][0] + merged_bounding_boxes[-1][2] + 20:
                    merged_bounding_boxes.append([x, y, w, h])
                else:
                    merged_bounding_boxes[-1][2] = x - merged_bounding_boxes[-1][0] + w
                    merged_bounding_boxes[-1][1] = min(merged_bounding_boxes[-1][1], y)
                    merged_bounding_boxes[-1][3] = y - merged_bounding_boxes[-1][1] + h
            else:
                merged_bounding_boxes.append([x, y, w, h])
    # drop bounding boxes on edge if there are multiple
    if len(merged_bounding_boxes) > 1:
        x, y, w, h = merged_bounding_boxes[0]
        if x == 0:
            merged_bounding_boxes = merged_bounding_boxes[1:]
            # createOverlap(greyImg, merged_bounding_boxes[0])
        x, y, w, h = merged_bounding_boxes[-1]
        if x+w >= greyImg.shape[1]:
            merged_bounding_boxes = merged_bounding_boxes[:-1]
            # createOverlap(greyImg, merged_bounding_boxes[0])

    # drop small bounding boxes
    traffic_signs = []
    for b in range(len(merged_bounding_boxes)):
        x, y, w, h = merged_bounding_boxes[b]
        if w > 10 and h > 5:
            final_bounding_boxes.append(merged_bounding_boxes[b])
            traffic_signs.append(cv2.resize(greyImg[y:y + h, x:x + w], (32, 20)))

    if len(traffic_signs) == 0:
        return greyImg[0:20, 0:32], None
    return traffic_signs[0], final_bounding_boxes[0]

def createOverlap(img, bound):
    plt.clf()
    plt.imshow(img)
    curAxis = plt.gca()

    curAxis.axis('off')
    x, y, w, h = bound
    coords = (x, y), w, h
    curAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor='red', linewidth=3))
    plt.show()

def consecutiveMax(data, stepsize=1):
    dataSplit = numpy.split(data, numpy.where(numpy.diff(data) > stepsize)[0]+1)
    maxlen = 0
    retData = None
    for d in dataSplit:
        if d.size > maxlen:
            maxlen = d.size
            retData = d
    return retData

if __name__ == "__main__":
    # read input image
    modelTrain = trainTrafficSign()
    modelTrain.loadData()
    modelTrain.train()
    modelTrain.save()

    print("done")
