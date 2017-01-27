import numpy as np
from osgeo import gdal
from osgeo import ogr
import scipy
from scipy import misc
import osr
import json

gdal.UseExceptions()
gdal.AllRegister()

def getAttr(B):
	row=B.RasterXSize
	col=B.RasterYSize
	nbands=B.RasterCount
	return col,row,nbands

def give_raster(filename):
	f=gdal.Open(filename)
	[col,row,nbands]=getAttr(f)
	RI=np.zeros((nbands,row,col))
	for i in range(0,nbands):
		np.copyto(RI[i],np.transpose(f.GetRasterBand(i+1).ReadAsArray(0,0,row,col)))
	return RI
	
def create_window(raster,posx,posy,sx,sy):
	band=[]
	[nbands,row,col]=raster.shape
	
	x_min = posx-int(sx/2)
	y_min = posy-int(sy/2)
	x_max = posx+int(sx/2)
	y_max = posy+int(sy/2)
	
	window=np.zeros((nbands,sx,sy))
	#print nbands
	for k in range(0,nbands):
		w=np.zeros((sx,sy))	
		for i in range(x_min,x_max+1):
			for j in range(y_min,y_max+1):
				if(i>=0 and i<row and j>=0 and j<col):
					w[i-x_min][j-y_min]=raster[k][i][j]
		np.copyto(window[k],w)
	return window

def get_chunk(I,px,py,wx,wy):
	shp=I.shape
	assert (px+wx)<shp[1]
	assert (py+wy)<shp[2]
	
	C=I[:,px:px+wx,py:py+wy]
	return C

def normalize(X):
	if(max(X)+min(X)==0):
		return X
	return (X-min(X))/(max(X)+min(X))
	
def getlabels(path,posx,posy):
	w=create_window(path,posx,posy,1,1)/255
	return max(w[0:3])

def pixel2coord(ds,col,row):
	c, a, b, f, d, e = ds.GetGeoTransform()
	xp = a * col + b * row + a * 0.5 + b * 0.5 + c
	yp = d * col + e * row + d * 0.5 + e * 0.5 + f
	return xp,yp

def array_to_raster(array, dst_filename, SourceDS):
	x_pixels = array.shape[1]  # number of pixels in x
	y_pixels = array.shape[0]  # number of pixels in y
	GeoT = SourceDS.GetGeoTransform()
	Projection = osr.SpatialReference()
	Projection.ImportFromWkt(SourceDS.GetProjectionRef())
	
	driver = gdal.GetDriverByName('GTiff')

	dataset = driver.Create(dst_filename,x_pixels,y_pixels,1,gdal.GDT_Float32)

	dataset.SetGeoTransform(GeoT)  

	dataset.SetProjection(Projection.ExportToWkt())
	dataset.GetRasterBand(1).WriteArray(array)
	dataset.FlushCache() 
	return
	

def getAllPoints(fname):
	driver = ogr.GetDriverByName('ESRI Shapefile')
	dataSource = driver.Open(fname, 0)
	layer=dataSource.GetLayer()
	layer.ResetReading()
	pts=[]
	for feature in layer:
		geom=feature.GetGeometryRef()
		js=geom.ExportToJson()
		jsobj=json.loads(js)
		for i in range(0,len(jsobj['coordinates'])):
			pts.append(jsobj['coordinates'][i])
	return np.array(pts)
		
		

#a=np.zeros((100,100))

#for i in range(0,100):
#	for j in range(0,100):
#		a[i][j]=getlabels('/home/ubuntu/workplace/saptarshi/codes/dcap/classified/classified_img1991.tif',i+100,j+100)

#scipy.misc.imsave('test.pbm',a)








