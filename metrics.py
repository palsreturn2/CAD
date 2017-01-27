import numpy as np

def change_metric(R,Bt,Btnxt, Btpred):
	shp = Bt.shape
	A=0
	B=0
	C=0
	D=0
	E=0
	for i in range(0,shp[0]):
		for j in range(0,shp[1]):
			if(Bt[i][j]-Btnxt[i][j]!=0):
				if(Btpred[i][j]-Bt[i][j]!=0):
					B=B+1
				else:
					A=A+1
			if(Bt[i][j]-Btnxt[i][j]==0):
				if(Btpred[i][j]-Bt[i][j]!=0):
					D=D+1
				else:
					E=E+1
			if(Btpred[i][j]!=Btnxt[i][j]):
				C=C+1
	print A,B,C,D,E
	return float(B)/(A+B+C+D), float(B)/(A+B+C), float(B)/(B+C+D), float(B+E)/(A+B+C+D+E)

