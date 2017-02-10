import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from matplotlib import cm
import sklearn.tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import time

def method_fit(trX,trY,B):
	pca=PCA(n_components=5)
	trY[np.logical_and(trY>=0, B<0)] = 1
	trY[np.logical_and(trY>=0, B>=0)] = 2
	trY[np.logical_and(trY<=0, B<0)] = 0

	model = SGDClassifier()
	#model = DecisionTreeClassifier()
	#model = RandomForestClassifier()
	#model = MLPClassifier()
	#model = AdaBoostClassifier()
	#model = SVC()
	#model = GaussianNB()
	start = time.time()
	model.fit(trX,trY)
	print time.time()-start
	#sklearn.tree.export_graphviz(model, out_file = 'ca_Decision_tree.dot', max_depth=5)
	#exit()
	return model

	
	
