# Machine Learning @ edX
# Week3 Project
# Renjie Li, rl2932@columbia.edu
import numpy
import sys

class project1(object):

    def __init__(self):
        self.lambda_input = float(sys.argv[1])
        self.sigma2_input = float(sys.argv[2])
        self.X_train = numpy.genfromtxt(sys.argv[3], delimiter=",")
        self.y_train = numpy.genfromtxt(sys.argv[4])
        self.X_test = numpy.genfromtxt(sys.argv[5], delimiter=",")
        self.row = self.X_train.shape[0] # number of rows
        self.col = self.X_train.shape[1] # number of columns
        self.wRR = self.ridge_regression()
        numpy.savetxt("wRR_" + str(int(self.lambda_input)) + ".csv", self.wRR, delimiter="\n")  # write output to file
        print('finish')

    def ridge_regression(self):
        id_matrix = self.lambda_input * numpy.identity(self.col)
        X_transpose = self.X_train.transpose()
        inv_matrix = numpy.linalg.inv(id_matrix + numpy.dot(X_transpose, self.X_train))
        wRR = numpy.dot(numpy.dot(inv_matrix, X_transpose), self.y_train)
        return wRR


if __name__ == "__main__":
    proj = project1()