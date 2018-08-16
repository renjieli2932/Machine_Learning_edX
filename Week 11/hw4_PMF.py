# Machine Learning @ edX
# Week11 Project
# Renjie Li, rl2932@columbia.edu
from __future__ import division
import numpy
import sys

class project4(object):

    def __init__(self):
        self.train_data = numpy.genfromtxt(sys.argv[1], delimiter=",")
        self.lam = 2
        self.sigma2 = 0.1
        self.d = 5
        self.PMF()
        print('finish')

    def PMF(self):
        self.Iteration = 50 # 50 Iterations
        self.Objective = numpy.zeros((self.Iteration,1))
        self.N1 = int(numpy.amax(self.train_data[:,0]))  # User numbers
        self.N2 = int(numpy.amax(self.train_data[:,1]))  # Object numbers
        self.u = numpy.random.normal(0,numpy.sqrt(1/self.lam),(self.N1,self.d)) # Generative Models
        self.v = numpy.random.normal(0,numpy.sqrt(1 /self.lam), (self.N2, self.d))
        self.M = numpy.zeros((self.N1,self.N2)) # Mij
        for data in self.train_data:
            self.M[int(data[0])-1,int(data[1])-1] = data[2]

        self.ui = []
        self.vj = []
        #print((self.train_data[self.train_data[:,0] == 1 ][:,1]).astype(int))
        for i in range(self.N1):
            objlist = self.train_data[self.train_data[:,0] == i+1 ][:,1].astype(int)
            self.ui.append(objlist)
        for j in range(self.N2):
            objlist = self.train_data[self.train_data[:, 1] == j + 1][:, 0].astype(int)
            self.vj.append(objlist)


        for iter in range(self.Iteration):
            # Update user location
            for i in range(self.N1):
                vjT = self.v[self.ui[i] -1 ]
                inbracket = numpy.linalg.inv(self.lam * self.sigma2 * numpy.identity(self.d) + numpy.dot(vjT.transpose(),vjT))
                Mij = self.M[i,self.ui[i]-1]
                sum = 0
                for k in range(len(Mij)):
                    sum += vjT[k] * Mij[k]
                self.u[i] = numpy.dot(inbracket,sum)
                #print(self.u[i])

            #Update object location
            for j in range(self.N2):
                uiT = self.u[self.vj[j] -1 ]
                inbracket = numpy.linalg.inv(self.lam * self.sigma2 * numpy.identity(self.d) + numpy.dot(uiT.transpose(), uiT))
                Mij = self.M[self.vj[j]-1,j]
                sum = 0
                for k in range(len(Mij)):
                    sum += uiT[k] * Mij[k]
                self.v[j] = numpy.dot(inbracket,sum)


            #a1 = ((numpy.linalg.norm(self.u, axis=1)))
            #a2 = ((numpy.linalg.norm(self.u, axis=1))**2).sum()

            # MAP Objective function
            sum = 0
            for data in self.train_data:
                sum += (data[2] - numpy.dot(self.u[int(data[0])-1],self.v[int(data[1]-1)])) ** 2
            sum = sum / (2.0 * self.sigma2)

            sum_ui = ((numpy.linalg.norm(self.u,axis=1) ** 2).sum()) * self.lam / 2.0
            sum_vj = ((numpy.linalg.norm(self.v, axis=1) ** 2).sum()) * self.lam / 2.0

            self.Objective[iter] = -sum - sum_ui - sum_vj

            if (iter == 9) or (iter == 24) or (iter == 49):
                #print(self.u[iter])
                Ufilename = "U-" + str(iter+1) +".csv"
                Vfilename = "V-" + str(iter+1) +".csv"
                numpy.savetxt(Ufilename, self.u, delimiter=",")
                numpy.savetxt(Vfilename, self.v, delimiter=",")

        numpy.savetxt("objective.csv", self.Objective, delimiter=",")





if __name__ == "__main__":
    proj = project4()