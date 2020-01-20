#--load the model from the disk--
import pickle
import numpy as np


filename='diabetes.sav'
loaded_model = pickle.load(open(filename,'rb'))

#--some predictions
Glucose = 65
BMI = 70
Age = 50

prediction = loaded_model.predict([[Glucose,BMI,Age]])
print(prediction)

if (prediction[0]==0):
    print("Non-diabetic")
else:
    print("Diabetic")


#-probability of the prediction
proba = loaded_model.predict_proba([[Glucose,BMI,Age]])
print(proba)
print("Confidence: " + str(round(np.amax(proba[0]) * 100 ,2)) + "%")