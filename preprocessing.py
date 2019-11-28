from model import UltraClassifier
import cv2
import matplotlib.pyplot as plt
import os

class Preprocessing(UltraClassifier):

    def preprocessing(self):
        path = "/Users/besshy/PycharmProjects/AIcontest/data/test/"
        file_name = os.listdir(path)
        if file_name[0] == ".DS_Store":
            self.img = cv2.imread(path + file_name[1])
        else:
            self.img = cv2.imread(path + file_name[0])
        self.img_up = self.img[0:700, 400:1100]
        self.height = self.img_up.shape[0]
        self.width = self.img_up.shape[1]
        ##print(self.img_up.shape)
        if file_name[0] == ".DS_Store":
            cv2.imwrite(path + file_name[1], self.img_up)
        else:
            cv2.imwrite(path + file_name[0], self.img_up)
        ##for Debag
        #plt.imshow(cv2.cvtColor(self.img_up, cv2.COLOR_BGR2RGB))
        #plt.show()
        return 0

if __name__ == "__main__":
    m = Preprocessing()
    m.preprocessing()