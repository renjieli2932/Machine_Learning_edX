# Machine Learning @ edX
# Week8 Project
# Renjie Li, rl2932@columbia.edu
import numpy
import sys


class project3(object):

    def __init__(self):
        self.X = numpy.genfromtxt(sys.argv[1], delimiter=",")
        self.Cluster = 5
        self.Iteration = 10
        self.row = self.X.shape[0] # Number of input
        self.col = self.X.shape[1] # Dimensions
        self.KMeans()
        self.EMGMM()

    def KMeans(self):
        # K-means Clustering
        c = numpy.zeros(self.row)  # Initialization of c

        init_mu = numpy.random.randint(0,self.row,size=self.Cluster) # Pick 5 random data points
        mu = self.X[init_mu]

        for iter in range(self.Iteration):
            # Update each c_i
            for i,xi in enumerate (self.X):
                inbracket = numpy.linalg.norm(xi-mu,ord=2,axis=1)
                c[i] = numpy.argmin(inbracket)

            # Update each mu_k
            nk = numpy.zeros(self.Cluster)  # Initialization of nk (nk = 1{Ci=k})
            for i in range(self.row):
                nk[c[i]] += 1

            for i in range(self.Cluster):
                sum = numpy.zeros(self.col)
                for j in range(self.row):
                    sum += self.X[j] * (c[j] == i)

                mu[i] = 1.0 * sum / nk[i]

            filename = "centroids-" + str(iter + 1) + ".csv"  # "i" would be each iteration
            numpy.savetxt(filename, mu, delimiter=",")


    def EMGMM(self):
        # EM for the GMM
        #Sigma = numpy.zeros((self.col,self.col,self.Cluster))
        iden = numpy.identity(self.col)
        Sigma = numpy.dstack([iden]*self.Cluster)
        pi = numpy.ones(self.Cluster) * 1.0 / self.Cluster # Uniform distribution
        Phi = numpy.zeros((self.row,self.Cluster))

        init_mu = numpy.random.randint(0,self.row,size=self.Cluster) # Pick 5 random data points
        mu = self.X[init_mu]

        for iter in range(self.Iteration):
            # E-step
            for k in range(self.Cluster):
                invSigma = numpy.linalg.inv(Sigma[:,:,k])
                detSigma = (numpy.linalg.det(Sigma[:,:,k])) ** (-0.5)

                for i in range(self.row):
                    xi = self.X[i]
                    inbracket = numpy.dot(numpy.dot((xi-mu[k]).transpose(),invSigma),(xi-mu[k]))
                    Phi[i,k] = pi[k] * ((2*numpy.pi)**(-self.col/2.0)) * detSigma * numpy.exp(-0.5*inbracket)

                for i in range(self.row):
                    sum = Phi[i].sum()
                    Phi[i] = 1.0 * Phi[i] / sum

            # M-step
            nk = numpy.sum(Phi,axis=0)
            pi = nk *1.0 / self.row

            for k in range(self.Cluster):
                mu[k] = numpy.dot(Phi[:,k].transpose(),self.X) * 1.0 / nk[k]

            for k in range(self.Cluster):
                inbracket = numpy.zeros(self.col)
                sum = numpy.zeros((self.col,self.col))

                for i in range(self.row):
                    xi = self.X[i]
                    inbracket = xi - mu[k]
                    sum += Phi[i,k] * numpy.dot(inbracket.transpose(),inbracket)

                Sigma[:,:,k] = sum *1.0 / nk[k]

            filename = "pi-" + str(iter + 1) + ".csv"
            numpy.savetxt(filename, pi, delimiter=",")
            filename = "mu-" + str(iter + 1) + ".csv"
            numpy.savetxt(filename, mu, delimiter=",")  # this must be done at every iteration

            for k in range(self.Cluster):
                filename = "Sigma-" + str(k + 1) + "-" + str(iter + 1) + ".csv" 
                numpy.savetxt(filename, Sigma[:,:,k], delimiter=",")



if __name__ == "__main__":
    proj = project3()
