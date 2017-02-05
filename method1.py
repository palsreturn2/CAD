import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from matplotlib import cm

def method_fit(trX,trY,B):
	pca=PCA(n_components=5)
	dv = pca.fit_transform(trX)
	trY[np.logical_and(trY>=0, B<0)] = 1
	trY[np.logical_and(trY>=0, B>=0)] = 2
	trY[np.logical_and(trY<=0, B<0)] = 0
	
	'''fig, ax = plt.subplots()

	cax = ax.scatter(dv[:,0],dv[:,1],marker='o',c=trY,cmap = cm.coolwarm)
	ax.set_title('Visualization of Feature vectors for Urban growth CA model')
	cbar = fig.colorbar(cax, ticks=[0, 1, 2])
	cbar.ax.set_yticklabels(['Non-Urban to Urban', 'Urban to Urban', 'Non-Urban to Non-Urban'])  # vertically oriented colorbar

	plt.show()'''

	model = SGDClassifier()
	#model = SVC()
	model.fit(dv,trY)
	
	return model

def method_predict(model,teX):
	pca = PCA(n_components = 5)
	dv = pca.fit_transform(teX)
	
	return model.predict(dv)
	
	
