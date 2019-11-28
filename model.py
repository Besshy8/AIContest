import cv2
import matplotlib.pyplot as plt
import numpy
import math
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPool2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input

class UltraClassifier():
    def __init__(self,num_train=100,num_val=20,batch_size=5,epoch=2):
        self.classes = ["BALTAN","METRON","TAIGA","ULTRA",]
        self.nb_classes = len(self.classes)
        self.nb_train_sample = num_train
        self.nb_validation_sample = num_val
        self.batch_size = batch_size
        self.n_epoch = epoch

    def preprocessing(self):
        for i in range(0,229):
            self.reshape_img(i)

    def reshape_img(self,num_img):

        if num_img <= 9:
            self.img = cv2.imread("./img/Job_0_000" + str(num_img) + "_Neutral.png")
        elif 9 < num_img < 100:
            self.img = cv2.imread("./img/Job_0_00" + str(num_img) + "_Neutral.png")
        else:
            self.img = cv2.imread("./img/Job_0_0" + str(num_img) + "_Neutral.png")

        self.img_up = self.img[0:700, 400:1100]
        """
        plt.imshow(cv2.cvtColor(self.img_up,cv2.COLOR_BGR2RGB))
        plt.show()
        """
        self.height = self.img_up.shape[0]
        self.width = self.img_up.shape[1]
        ##print(self.img_up.shape)
        cv2.imwrite("./img_std/Job_" + str(num_img) + "_Neutral.png",self.img_up)

        return 0

    ##CNN
    def modelCNN(self):
        self.model = Sequential()
        self.model.add(Conv2D(32,(3,3),padding="same",activation="relu",
                              input_shape=(224,224,3)))
        self.model.add(Conv2D(32,(3,3),padding="same",activation="relu"))
        self.model.add(MaxPool2D(pool_size=(2,2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        self.model.add(MaxPool2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(256, activation="relu"))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(4, activation="softmax"))
        self.model.summary()

        self.model.compile(loss='categorical_crossentropy',
                      optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
                      metrics=['accuracy'])

        return 0

    ##VGG16

    def load_img_itr(self,train_url,val_url):
        self.img_train = ImageDataGenerator(
            rescale=1/255.,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            preprocessing_function=preprocess_input
        )
        self.img_itr_train = self.img_train.flow_from_directory(
            train_url,
            target_size=(224, 224),
            color_mode='rgb',
            batch_size=5,
            classes=self.classes,
            class_mode='categorical',
            shuffle=True
        )

        self.img_itr_validation = self.img_train.flow_from_directory(
            val_url,
            target_size=(224, 224),
            color_mode='rgb',
            batch_size=5,
            classes=self.classes,
            class_mode='categorical',
            shuffle=True
        )
        return 0

    def modelVGG16(self):
        self.vgg16 = VGG16(include_top=False,input_shape=(224,224,3))
        self.vgg16.summary()
        self.bulid_transfer_model()
        self.model.summary()
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=SGD(lr=1e-4,momentum=0.9),metrics=['accuracy'])
        return 0

    def bulid_transfer_model(self):
        self.model = Sequential(self.vgg16.layers)
        for layer in self.model.layers[:15]:
            layer.trainable = False

        self.model.add(Flatten())
        self.model.add(Dense(256,activation="relu"))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(4,activation="softmax"))

        return 0

    def fit(self):
        steps_per_epoch = math.ceil(self.img_itr_train.samples / self.batch_size)
        validation_steps = math.ceil(self.img_itr_validation.samples / self.batch_size)

        history = self.model.fit_generator(
            self.img_itr_train,
            steps_per_epoch=steps_per_epoch,
            epochs=self.n_epoch,
            validation_data=self.img_itr_validation,
            validation_steps=validation_steps
        )
        return 0

    def evaluateModel(self):
        imsize = 224
        test = cv2.imread('data/test/Job_182_Neutral.png')
        test = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)
        ##plt.imshow(test)
        test = cv2.resize(test, (imsize, imsize))
        test = test.reshape((1, imsize, imsize, 3)) / 255.
        score = self.model.predict(test).reshape((4,))
        score *= 100
        print("BALTAN: %.2f%%, METRON: %.2f%%,TAIGA: %.2f%%,ULTRA: %.2f%%" % (
        score[0], score[1], score[2], score[3]))
        ##plt.show()
        return 0

    def save_model(self):
        open("./model_save/CNN.json","w").write(self.model.to_json())
        self.model.save_weights("./model_save/CNN.h5")
        return 0


if __name__ == "__main__":
    ##m.preprocessing()
    m = UltraClassifier(100,20,5,2)
    flag = 0
    if flag == 1:
        m.modelVGG16() ##Change model of VGG16 into model of Ultra AIContest(4 classifier)
        m.load_img_itr("./data/train","./data/validation")
        m.fit()
    else:
        m.modelCNN()
        m.load_img_itr("./data/train", "./data/validation")
        m.fit()
    m.save_model()
    m.evaluateModel()








