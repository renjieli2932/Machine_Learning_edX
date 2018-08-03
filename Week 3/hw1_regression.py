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
        self.ridge_regression() # Solution for part 1
        self.active_learning()  # Solution for part 2


    def ridge_regression(self):
        id_matrix = self.lambda_input * numpy.identity(self.col)
        X_transpose = self.X_train.transpose()
        inv_matrix = numpy.linalg.inv(id_matrix + numpy.dot(X_transpose, self.X_train))
        wRR = numpy.dot(numpy.dot(inv_matrix, X_transpose), self.y_train)
        numpy.savetxt("wRR_" + str(int(self.lambda_input)) + ".csv", wRR, delimiter="\n")  # write output to file

    def update(self):
        id_matrix = self.lambda_input * numpy.identity(self.col)
        X_transpose = self.X_train.transpose()
        temp1 = numpy.dot(X_transpose, self.X_train) + self.XXT
        self.Sigma = numpy.linalg.inv(id_matrix + 1.0 / self.sigma2_input * temp1)

        id_sigma_matrix = self.sigma2_input * id_matrix
        temp2 = numpy.dot(X_transpose,self.y_train) + self.XY
        self.Miu = id_sigma_matrix + numpy.dot(numpy.linalg.inv(temp1),temp2)



    def active_learning(self):
        self.XXT = numpy.zeros((self.col,self.col)) #Initialization for X0X0T update
        self.XY = numpy.zeros(self.col) #Initialization for X0Y0 update
        active_index = [] # Store the desired indices
        self.Sigma = 0
        self.Miu = 0
        self.update()  # Initialization for Posterior


        location = list(range(self.X_test.shape[0])) # locate the desired x0
        for i in range(10):
            sigma0_2 = numpy.zeros(self.X_test.shape[0])
            for j in range(self.X_test.shape[0]):
                sigma0_2[j] = self.sigma2_input + numpy.dot(numpy.dot(self.X_test[j],self.Sigma),self.X_test[j].transpose())
            print(sigma0_2)
            pick = numpy.argmax(sigma0_2)
            print(pick)
            x0 = self.X_test[pick]
            y0 = numpy.dot(x0,self.Miu)
            self.XXT += numpy.dot(x0.transpose(),x0)
            self.XY += numpy.dot(x0.transpose(),y0) # not sure here

            abs_location = location.pop(pick) + 1
            active_index.append(abs_location)
            self.X_test = numpy.delete(self.X_test,pick,0) # delete the max x0
            self.update()

        print(active_index)
        numpy.savetxt("active_" + str(int(self.lambda_input)) + "_" + str(int(self.sigma2_input)) + ".csv", active_index,delimiter=",")  # write output to file


if __name__ == "__main__":
    proj = project1()