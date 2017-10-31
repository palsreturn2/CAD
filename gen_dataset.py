from PIL import Image
import numpy as np
import geninput as INPUT
import math
from osgeo import gdal
from osgeo import ogr
from skimage import io
import scipy

def cad_dataset(R,Bt,Btnxt,wx,wy):	
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
			if(R[0][i][j]!=0):
				x=INPUT.create_window(Rn,i,j,wx,wy)			
				V.append(x)
	return np.array(trX),np.array(trY),np.array(teX),np.array(teY),np.array(V),np.array(B)

def func_populate_H(start_x, start_y, end_x, end_y, H, feature):
	if start_x>end_x:
		return
	
	if start_x==end_x:
		if start_y==end_y:
			if feature not in H[start_x][start_y]:
				H[start_x][start_y].append(feature)
			return H
		for y in range(start_y, end_y):
			if feature not in H[start_x][y]:
				H[start_x][y].append(feature)
		return H	
	
	deltax = end_x-start_x
	deltay = end_y-start_y
	deltaerr = 2*deltay-deltax
	y = start_y
	for x in range(start_x, end_x):
		if feature not in H[x][y]:
			H[x][y].append(feature)
		while(deltaerr>0):
			y=y+1
			if feature not in H[x][y]:
				H[x][y].append(feature)
			deltaerr = deltaerr - 2*deltax
			
		deltaerr = deltaerr + 2*deltay
	return H
	
def cad_dataset_vector(raster_ds ,vector_ds , wx, wy):
	[col, row, nbands] = INPUT.getAttr(raster_ds)
	R = np.zeros((nbands,row,col))
	H = [[[] for i in range(0,col)] for k in range(0,row)]
	for i in range(0,nbands):
		np.copyto(R[i],np.transpose(raster_ds.GetRasterBand(i+1).ReadAsArray(0,0,row,col)))
	X = []
	layer = vector_ds.GetLayer()
	layer.ResetReading()
	I = np.zeros([row,col])
	c = 0
	for feature in layer:
		geom = feature.GetGeometryRef()
		if geom.GetGeometryCount()==0:
			for i in range(0,geom.GetPointCount()-1):
				start_row, start_col = INPUT.coord2pixel(raster_ds, geom.GetPoint(i)[0], geom.GetPoint(i)[1])
				end_row, end_col = INPUT.coord2pixel(raster_ds, geom.GetPoint(i+1)[0], geom.GetPoint(i+1)[1])
				if start_row<=end_row:
					H = func_populate_H(start_row, start_col, end_row, end_col, H, feature)
				elif start_row>end_row:
					H = func_populate_H(end_row, end_col, start_row, start_col, H, feature)
					
		for j in range(0, geom.GetGeometryCount()):
			g = geom.GetGeometryRef(j)
			
			if g.GetPointCount()==1:
				start_row, start_col = INPUT.coord2pixel(raster_ds, g.GetPoint(0)[0], g.GetPoint(0)[1])
				I[start_row][start_col] = 255
				if feature not in H[start_row][start_col]:
					H[start_row][start_col].append(feature)
			for i in range(0,g.GetPointCount()-1):
				
				start_row, start_col = INPUT.coord2pixel(raster_ds, g.GetPoint(i)[0], g.GetPoint(i)[1])
				end_row, end_col = INPUT.coord2pixel(raster_ds, g.GetPoint(i+1)[0], g.GetPoint(i+1)[1])
				
					
				if start_row<=end_row:
					H = func_populate_H(start_row, start_col, end_row, end_col, H, feature)
				elif start_row>end_row:
					H = func_populate_H(end_row, end_col, start_row, start_col, H, feature)

	for i in range(0,row):
		for j in range(0,col):
			if len(H[i][j])!=0:
				I[i][j] = 1
				c = c+1
	io.imsave('./dataset/debug.png', np.transpose(I))
	c = 0 
	for i in range(0,row):
		for j in range(0,col):
			if(R[0][i][j]!=0):				
				ml = INPUT.create_vector_window(raster_ds, i, j, wx, wy, H)
				if ml.ExportToWkt()!='MULTILINESTRING EMPTY':
					#print "not empty"
					c=c+1
				X.append(ml)
	print c
	return np.array(X)

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
	roads_loc = '/home/ubuntu/workplace/saptarshi/Data/roads/mumbai/'
	raster_loc = '/home/ubuntu/workplace/saptarshi/Data/raw/mumbai/'
	
	driver = ogr.GetDriverByName('ESRI Shapefile')

	vector_ds = driver.Open(roads_loc+'roadsM.shp',0)
	raster_ds = gdal.Open(raster_loc+'1991.tif')
	X = cad_dataset_vector(raster_ds, vector_ds, 1,1)
	np.save('./dataset/Road_trX.npy',X)
		
	
	
	
	
	
