from sklearn import datasets
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn import LogisticRegression

digits=datasets.load_digits()

#print(digits.data[1796],digits.data[1796].shape)
#print(digits.target[1796])
#print(digits.images[1])

def digittoshow(inp):
    clf=svm.SVC(gamma=.001,C=100)
    x,y=digits.data[:-1],digits.target[:-1]

    #predict
    clf.fit(x,y)
    print("Prediction : ",clf.predict([digits.data[inp]]))

#display the image
    plt.imshow(digits.images[inp],cmap=plt.cm.gray_r,interpolation="nearest")
    plt.show()

digittoshow(8)