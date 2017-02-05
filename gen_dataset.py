from PIL import Image
import numpy as np
import geninput as INPUT
import math
from osgeo import gdal
from skimage import io

def dcap_dataset(R,Bt,Btnxt,wx,wy):	
	shp=R.shape
	Rn = R/(np.max(np.ndarray.flatten(R)))
	for i in range(0,shp[0]):
		Rn[i] = Rn[i] - np.mean(np.ndarray.flatten(Rn[i]))
	Rn=np.concatenate([Rn,Bt.reshape((1,shp[1],shp[2]))])
	trX = [] 
	trY = []
	teX = []
	teY = []
	V=[]
	B=[]
	for i in range(0,shp[1]):
		for j in range(0,shp[2]):
			if(R[0][i][j]!=0):
				x=INPUT.create_window(Rn,i,j,wx,wy)				
				trX.append(x)
				trY.append(Btnxt[i][j])
				B.append(Bt[i][j])
				V.append(x)
	return np.array(trX),np.array(trY),np.array(teX),np.array(teY),np.array(V),np.array(B)
	
def ageBuiltUp(R, Bt, Btnxt, age, first = True):
	shp = Bt.shape
	if(first):		
		Bt[Bt==0] = -1
		Bt = Bt*age
	for i in range(0,shp[0]):
		for j in range(0,shp[1]):
			if(R[0][i][j]!=0):
				if(Bt[i][j]>0 and Btnxt[i][j]>0):
					Btnxt[i][j] = Bt[i][j]+age
				if(Bt[i][j]>0 and Btnxt[i][j]<=0):
					Btnxt[i][j] = 0
				if(Bt[i][j]<=0 and Btnxt[i][j]<=0):
					Btnxt[i][j] = Bt[i][j]-age
				if(Bt[i][j]<=0 and Btnxt[i][j]>0):
					Btnxt[i][j] = age
	return Bt,Btnxt

def create_test_dataset(R,Bt,Btnxt):
	V=[]
	shp = R.shape
	Rn = R/(np.max(np.ndarray.flatten(R)))
	for i in range(0,shp[0]):
		Rn[i] = Rn[i] - np.mean(np.ndarray.flatten(Rn[i]))
	Rn=np.concatenate([Rn,Bt.reshape((1,shp[1],shp[2]))])
	for i in range(0,shp[1]):
		for j in range(0,shp[2]):
			if(R[0][i][j]!=0):
				x=INPUT.create_window(Rn,i,j,3,3)					
				V.append(x)
	return np.array(V)

def NearestNeighbors(k, case, D):
	N = []
	Dm = np.sqrt(np.sum((case - D)**2,axis=1))
	Dm = Dm[Dm>0]
	return np.argsort(Dm)[:k]

def genSyntheticCases(D,p,k):
	newCases = []
	ng = int(p*D.shape[0]/100)
	print D.shape[0]
	print ng
	for case in D:
		nns = NearestNeighbors(k,case,D)
		new = np.array([])
		for i in range(0,ng):
			x = D[nns[np.random.randint(low=0,high=k)]]
			
			diff = case - x
			new = case + np.random.randn(1)*diff
		
			d1 = math.sqrt(np.sum((new-case)**2))
			d2 = math.sqrt(np.sum((new-x)**2))
			new = (d1*case+d2*x)/(d1+d2)
			newCases.append(new)
	return np.array(newCases)
	
def smoter(X,Y,tE,p,u,k):
	ybar = np.mean(Y)	
	rareL = np.logical_and(Y<ybar, np.absolute(Y-X[:,31])>=tE)
	newCasesL = genSyntheticCases(X[rareL],p,k)
	rareH = np.logical_and(Y>ybar, np.absolute(Y-X[:,31])>=tE)
	newCasesH = genSyntheticCases(X[rareH],p,k)
	newCases = np.concatenate((newCasesH,newCasesL))
	nrnorm = u*newCases.shape[0]/100
	normCases = np.logical_not(np.logical_or(rareH,rareL))
	normCases[np.random.randint(low=0, high = normCases.shape[0], size = normCases.shape[0]-nrnorm)] = False
	return np.concatenate((normCases,newCases))
	
	
if __name__ == "__main__":
	trX = np.load('./dataset/DCAP_trX.npy')
	trY = np.load('./dataset/DCAP_trY.npy')
	trX = trX.reshape([-1,36])
	ntrX = smoter(trX,trY,0.1,0.0020,0.0020,3)
	print ntrX.shape
	np.save('./dataset/DCAP_SMOTE_trX.npy',ntrX)
	
	
	
