import numpy as np

def indexJaccard(py_x,testY):
	py_x=np.ndarray.flatten(py_x)
	testY=np.ndarray.flatten(testY)
	tp=0
	fp=0
	for k in range(0,len(py_x)):
		if(int(py_x[k])==int(testY[k])):
			tp=tp+1
		else:
			fp=fp+1
	return float(tp)/(tp+fp)

def rsquared(py_x,testY):
	l=py_x.shape[0]
	m=np.mean(testY,axis=0)
	print m
	ss_res=np.zeros(py_x.shape[1])
	ss_tot=np.zeros(py_x.shape[1])
	for k in range(0,l):
		ss_res=ss_res+((py_x[k]-testY[k])**2)
		ss_tot=ss_tot+((testY[k]-m)**2)
	c=[n/d for n,d in zip(ss_res,ss_tot) if d !=0]
	print c
	return np.mean(c)

def new_index(py_x,testY):
	l=py_x.shape[0]
	c=np.zeros(py_x.shape[1])
	for k in range(0,l-1):
		dpy=py_x[k+1]-py_x[k]
		dty=testY[k+1]-testY[k]		
		
		for i in range(0,py_x.shape[1]):
			if(dpy[i]==dty[i]):
				c[i]=c[i]+1
		acc=np.mean(c/l)
	return acc
	
def indexJ(P,T,labels):
	P=np.ndarray.flatten(P)
	T=np.ndarray.flatten(T)
	shp=P.shape
	c=0
	mc=0
	for i in range(0,shp[0]):
		if(int(T[i]) in labels):
			if(int(P[i])==int(T[i])):
				c=c+1
			else:
				mc=mc+1
	return float(c)/(mc+c)
		
