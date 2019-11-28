from keras.models import model_from_json
import cv2
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from display import display
import os
from model import UltraClassifier

class Predict():

    def evaluateModel(self):
        imsize = 224
        path = "/Users/besshy/PycharmProjects/AIcontest/data/test"
        file_name = os.listdir(path)
        if file_name[0] == ".DS_Store":
            test = cv2.imread('data/test/' + file_name[1])
        else:
            test = cv2.imread('data/test/' + file_name[0])
        test = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)
        ##plt.imshow(test)
        test = cv2.resize(test, (imsize, imsize))
        test = test.reshape((1, imsize, imsize, 3)) / 255.
        self.score =model.predict(test).reshape((4,))
        self.score *= 100
        print("=========================")
        print("BALTAN: %.2f%%, METRON: %.2f%%,TAIGA: %.2f%%,ULTRA: %.2f%%" % (
            self.score[0], self.score[1], self.score[2], self.score[3]))
        ##plt.show()
        return 0

    def result(self):
        self.dict = {self.score[0]:"BALTAN", self.score[1]: "METRON",
                     self.score[2]:"TAIGA", self.score[3]: "ULTRA"}
        self.prob = max(self.score)
        self.name = self.dict[self.prob]

        return self.prob,self.name

if __name__ == "__main__":
    model = model_from_json(open("model_save/CNN.json","r").read())
    model.load_weights("model_save/CNN.h5")
    pred = Predict()
    pred.evaluateModel()
    prob,name =pred.result()

    display(name,prob)