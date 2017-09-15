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
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import time
from sklearn.externals import joblib

def select_classifier(trX,trY):
	models = []
	models.append(MLPClassifier(hidden_layer_sizes=(10,), activation = "tanh", learning_rate='adaptive', max_iter=100)) 
	models.append(MLPClassifier(hidden_layer_sizes=(20,15), activation = "tanh", learning_rate='adaptive', max_iter=100)) 
	models.append(MLPClassifier(hidden_layer_sizes=(20,15,10), activation = "tanh", learning_rate='adaptive', max_iter=100)) 
	models.append(MLPClassifier(hidden_layer_sizes=(20,15,10,5), activation = "tanh", learning_rate='adaptive', max_iter=100)) 
	models.append(MLPClassifier(hidden_layer_sizes=(20,15,10,5,3), activation = "tanh", learning_rate='adaptive', max_iter=100)) 
	print "Model evaluation started"
	for model in models:
		scores = cross_val_score(model,trX,trY,cv=5)
		print("Accuracy: %f (+/- %f)" % (scores.mean(), scores.std() * 2))
	
	print "Model evaluation ended"

def method_fit(trX,trY,B, model):
	#pca=PCA(n_components=5)
	#print np.sum(np.logical_and(trY<=0, B>0))
	
	if(model == "sgd"):
		model = SGDClassifier()
	elif(model=="dt"):
		model = DecisionTreeClassifier()
	elif(model=="rf"):
		model = RandomForestClassifier(n_estimators=10)
	elif(model=="mlp"):
		model = MLPClassifier(hidden_layer_sizes=(100,), activation = "tanh", learning_rate='adaptive', max_iter=100)
	#model = MLPClassifier(activation = "tanh", learning_rate='adaptive', max_iter=100)
	elif(model=="ada"):
		model = AdaBoostClassifier()
	elif(model=="knn"):
		model = KNeighborsClassifier()
	elif(model=="svc"):
		model = SVC(kernel = "linear", tol=0.01, cache=7000)
	elif(model=="nb"):
		model = GaussianNB()
	
	#scores = cross_val_score(model,trX,trY,cv=2)
	#print("Accuracy: %f (+/- %f)" % (scores.mean(), scores.std() * 2))
	
	#print model
	start = time.time()
	model.fit(trX,trY)
	joblib.dump(model, 'model.pkl')
	print time.time()-start
	#exit()
	sklearn.tree.export_graphviz(model, out_file = 'ca_Decision_tree.dot', max_depth=None)
	return model

	
	
