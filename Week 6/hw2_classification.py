# Machine Learning @ edX
# Week6 Project
# Renjie Li, rl2932@columbia.edu
from __future__ import division
import numpy
import sys


class project2(object):

    def __init__(self):
        self.X_train = numpy.genfromtxt(sys.argv[1], delimiter=",")
        self.y_train = numpy.genfromtxt(sys.argv[2])
        self.X_test = numpy.genfromtxt(sys.argv[3], delimiter=",")
        self.row = self.X_train.shape[0]
        self.col = self.X_train.shape[1]
        self.K = 10
        self.ClassProcessing()
        self.pluginClassifier()



    def ClassProcessing(self):
        # Class Priors
        self.piy = 1.0 * numpy.unique(self.y_train,return_counts=True)[1] / self.row

        # Class Conditional Density
        self.Miu = numpy.zeros((self.K,self.col)) # Initialization
        self.Sigma = numpy.zeros((self.K,self.col,self.col)) #Initialization

        for i in range(self.K):
            xi= self.X_train[self.y_train == i]
            self.Miu[i] = numpy.mean(xi,axis = 0)
            self.Sigma[i] = numpy.dot((xi-self.Miu[i]).transpose(),xi-self.Miu[i]) * 1.0 / len(xi)


    def pluginClassifier(self):
        # Plug-in Classifier
        testrow = self.X_test.shape[0]
        self.Plugin = numpy.zeros((testrow,self.K))

        for i in range(self.K):
            SigmaDet = (numpy.linalg.det(self.Sigma[i])) ** (-0.5)
            invSigma = numpy.linalg.inv(self.Sigma[i])
            for j in range(testrow):
                inbracket = (-0.5) * numpy.dot(numpy.dot((self.X_test[j]-self.Miu[i]),invSigma),(self.X_test[j]-self.Miu[i]).transpose())
                self.Plugin[j,i] = self.piy[i] * SigmaDet * numpy.exp(inbracket)

        for i in range(testrow):
            sum = numpy.sum(self.Plugin[i])
            self.Plugin[i] = self.Plugin[i] * 1.0 / sum


        numpy.savetxt("probs_test.csv", self.Plugin, delimiter=",")  # write output to file



if __name__ == "__main__":
    proj = project2()
