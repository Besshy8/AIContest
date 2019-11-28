
## when predict , please call this method
import os

def display(name,prob):
    path = "/Users/besshy/PycharmProjects/AIcontest/data/test"
    file_name = os.listdir(path)
    ##print(file_name)
    print("=====RESULT=====")
    if file_name[0] == ".DS_Store":
        print("file name : %s" % file_name[1])
    else:
        print("file name : %s" % file_name[0])
    print("name : %s" % name)
    print("Probability: %f " % prob + "%")
    print("=======TIME(Predict)======")

