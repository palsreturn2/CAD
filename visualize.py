import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE 
from sklearn.decomposition import PCA
import sys
from matplotlib import cm
import geninput as INPUT

def viz1(R,f):
	shp = R.shape
	I = np.zeros([shp[1],shp[2]])
	k=0
	for i in range(0,shp[1]):
		for j in range(0,shp[2]):
			if(R[0][i][j]!=0):
				I[i][j]=f[k]
				k=k+1
	plt.imshow(np.transpose(I))
	plt.show()

if __name__ == "__main__":
	trX = np.load('./dataset/DCAP_trX.npy')
	trY = np.load('./dataset/DCAP_trY.npy')
	B = np.load('./dataset/DCAP_B.npy')
	R = INPUT.give_raster('/home/ubuntu/workplace/saptarshi/Data/raw/mumbai/1990.tif')
	trX = trX.reshape([-1,36])

	c1 = trX[np.logical_and(trY>=0, B<0)]
	c2 = trX[np.logical_and(trY<0, B>=0)]
	c3 = trX[np.logical_and(trY>=0, B>=0)]
	c4 = trX[np.logical_and(trY<=0, B<0)]

	#nf = int(sys.argv[2])

	v1 = c1
	v2 = c2
	v3 = c3
	v4 = c4

	v = np.concatenate([v1,v2,v3,v4])

	y = np.concatenate([np.zeros(v1.shape[0]),np.ones(v3.shape[0]),2*np.ones(v4.shape[0])])

	if(sys.argv[1] == 'pca'):
		pca=PCA(n_components=2)
		dv = pca.fit_transform(v)
		viz1(R,dv[:,0])
		viz1(R,dv[:,0])
	else:
		tsne = TSNE(n_components=2,random_state=0)
		dv = tsne.fit_transform(v)

	fig, ax = plt.subplots()

	cax = ax.scatter(dv[:,0],dv[:,1],marker='o',c=y,cmap = cm.coolwarm)
	ax.set_title('Visualization of Feature vectors for Urban growth CA model')
	cbar = fig.colorbar(cax, ticks=[0, 1, 2])
	cbar.ax.set_yticklabels(['Non-Urban to Urban', 'Urban to Urban', 'Non-Urban to Non-Urban'])  # vertically oriented colorbar

	plt.show()

