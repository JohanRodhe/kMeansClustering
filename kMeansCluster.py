import random
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
import pandas as pd

def load_iris():
	data = pd.read_csv(filepath_or_buffer='iris.data',  header=None,  sep=',')
	data.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
	data.dropna(how="all", inplace=True)
	data.tail()
	feature_matrix = data.ix[:,0:4].values
	labels = data.ix[:, 4].values

	return feature_matrix, labels

def load_data():
	mu1 = [-1, 1]
	sigma1 = [[1, 0.6], [0.6, 1.1]]
	mu2 = [1, -1]
	sigma2 = [[0.3, -0.2], [-0.2, 1]]
	x1 = np.random.multivariate_normal(mu1, sigma1, 75)
	x2 = np.random.multivariate_normal(mu2, sigma2, 150)
	return x1, x2

def cluster(X, mu, K):
	clusters = {}
	R = np.zeros(shape=(len(X), K))
	j = 0
	n = 0
	for x in X:
		best_mu = min([(i[0], la.norm(x - mu[i[0]])) for i in enumerate(mu)], key=lambda t:t[1])[0]
		for k in range(K):
			if best_mu == k:
				R[n][k] = 1
			else:
				R[n][k] = 0
		try:
			clusters[best_mu].append(x)
		except KeyError:
			clusters[best_mu] = [x]
		n += 1
	for n in range(len(X)):
		for i in range(K):
			j += R[n][i] * la.norm(X[n] - mu[i])
	return clusters, R, j

def update_prototypes(mu, clusters):
	new_mu = []
	keys = sorted(clusters.keys())
	for k in keys:
		new_mu.append(np.mean(clusters[k], axis=0))
	return new_mu

def kMeansConv(X, K):
	J = []
	old_mu = random.sample(X, K)
	mu = random.sample(X, K)
	while not converged(mu, old_mu):
		old_mu = mu
		clusters, R, j = cluster(X, mu, K)
		J.append(j)
		mu = update_prototypes(old_mu, clusters)

	return mu, clusters, R, J

def kMeansIter(X, K, Niter):
	J = []
	old_mu = random.sample(X, K)
	mu = random.sample(X, K)
	while(Niter >= 0):
		old_mu = mu
		clusters, R, j = cluster(X, mu, K)
		J.append(j)
		mu = update_prototypes(old_mu, clusters)
		Niter -= 1
	return mu, clusters, R, J



def converged(mu, oldmu):
	return (set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu]))



def main():
	K = 2
	Niter = 5
	Y = []
	x1, x2 = load_data()
	X = np.concatenate((x1, x2), axis=0)
	ratios = []
	mu, C, R, J = kMeansConv(X, K)
	for i in range(K):
		C[i] = np.mat(C[i])
		Y.append(C[i])
		ratios.append(np.sum(R[:,i]) / 225)
	print ratios
	plot(1, K, C, J)

def confTables():
	y_pred = []

	for i in range(150):
		if R[i,0] == 1:
			y_pred.append('Iris-setosa')
		if R[i,1] == 1:
			y_pred.append('Iris-versicolor')
		if R[i,2] == 1:
			y_pred.append('Iris-virginica')	
	y_actual = []
	for i in range(50):
		y_actual.append("Iris-setosa")
	for i in range(50):
		y_actual.append("Iris-versicolor")
	for i in range(50):
		y_actual.append("Iris-virginica")

	y_a = pd.Series(y_actual, name='Actual')
	y_p = pd.Series(y_pred, name='Predicted')

	df_confusion = pd.crosstab(y_a,y_p)
	plot_confusion_matrix(df_confusion)
	print df_confusion

def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
    plt.matshow(df_confusion, cmap=cmap) # imshow
    #plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    #plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)
    plt.show()


def plot2(C):
	cluster1, = plt.plot(C[0][:,0], C[0][:,1], 'ro')
	cluster2, = plt.plot(C[1][:,0], C[1][:,1], 'go')
	plt.legend([cluster1, cluster2], ["Cluster 1", "Cluster 2"])
	plt.show()

def plot3(C):
	cluster1, = plt.plot(C[0][:,0], C[0][:,1], 'ro')
	cluster2, = plt.plot(C[1][:,0], C[1][:,1], 'go')
	cluster3, = plt.plot(C[2][:,0], C[2][:,1], 'bo')
	plt.legend([cluster1, cluster2, cluster3], ["Cluster 1", "Cluster 2", "Cluster 3"])
	plt.show()

def plot4(C):
	cluster1, = plt.plot(C[0][:,0], C[0][:,1], 'ro')
	cluster2, = plt.plot(C[1][:,0], C[1][:,1], 'go')
	cluster3, = plt.plot(C[2][:,0], C[2][:,1], 'bo')
	cluster4, = plt.plot(C[3][:,0], C[3][:,1], 'yo')
	plt.legend([cluster1, cluster2, cluster3, cluster4], ["Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4"])
	plt.show()

def plot5(C):
	cluster1, = plt.plot(C[0][:,0], C[0][:,1], 'ro')
	cluster2, = plt.plot(C[1][:,0], C[1][:,1], 'go')
	cluster3, = plt.plot(C[2][:,0], C[2][:,1], 'bo')
	cluster4, = plt.plot(C[3][:,0], C[3][:,1], 'yo')
	cluster5, = plt.plot(C[4][:,0], C[4][:,1], 'ko')
	plt.legend([cluster1, cluster2, cluster3, cluster4, cluster5], ["Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4", "Cluster 5"])
	plt.show()

def plot(k, K, C, J):
	if k == 0:
		if(K == 2):
			plot2(C)
		if(K == 3):
			plot3(C)			
		if(K == 4):
			plot4(C)
		if(K == 5):
			plot5(C)

	if k == 1:
		plt.plot(J)
		plt.xlabel('Iterations')
		plt.ylabel('J')
		plt.plot(J, 'ro')
		plt.show()

if __name__ == "__main__":
	main()