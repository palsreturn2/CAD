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
	
def ageBuiltUp(R, Bt, Btnxt, age):
	shp = Bt.shape
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
