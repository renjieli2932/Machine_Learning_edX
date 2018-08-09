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
        pass


'''
def KMeans(data):
    # perform the algorithm with 5 clusters and 10 iterations...you may try others for testing purposes, but submit 5 and 10 respectively

    filename = "centroids-" + str(i + 1) + ".csv"  # "i" would be each iteration
    np.savetxt(filename, centerslist, delimiter=",")


def EMGMM(data):
    filename = "pi-" + str(i + 1) + ".csv"
    np.savetxt(filename, pi, delimiter=",")
    filename = "mu-" + str(i + 1) + ".csv"
    np.savetxt(filename, mu, delimiter=",")  # this must be done at every iteration


for j in range(k):  # k is the number of clusters
    filename = "Sigma-" + str(j + 1) + "-" + str(i + 1) + ".csv"  # this must be done 5 times (or the number of clusters) for each iteration
    np.savetxt(filename, sigma[j], delimiter=",")
'''



if __name__ == "__main__":
    proj = project3()
