from sklearn.svm import SVC
import numpy as np
import cPickle
import os
model=None
def load(filename):
	__location__ = os.path.realpath(
		os.path.join(os.getcwd(), os.path.dirname(__file__)))
	path=os.path.join(__location__, 'my_dumped_classifier.pkl')
	#print(path + "  checking \n\n")
	with open(path, 'rb') as fid:
    	 return cPickle.load(fid)

# Input should be of the form
# Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome	
def checkDiabetes(a,b,c,d,e,f,g,h):
	input=getNumpyArrayFromInput(a,b,c,d,e,f,g,h)
	global model
	model=load('my_dumped_classifier.pkl')
	response = model.predict(input)
	return response


#print checkDiabetes(np.array([6,148,72,35,0,33.6,0.627,50]).reshape(1,-1));

# 8 values as parameters input
# you should have got these parameters from request body
def getNumpyArrayFromInput(a,b,c,d,e,f,g,h):
	a=float(a);b=float(b);c=float(c);d=float(d);e=float(e);f=float(f);g=float(g);h=float(h);
	return np.array([a,b,c,d,e,f,g,h]).reshape(1,-1)


