from flask import Flask, render_template, url_for, request, redirect
from flask_sqlalchemy import SQLAlchemy
import numpy as np
import pickle



filename='diabetes.sav'
loaded_model = pickle.load(open(filename,'rb'))

app = Flask(__name__)



@app.route('/',methods=['POST','GET'])
def index():
    if request.method=='POST':
        Glucose = request.form['Glucose']
        BMI = request.form['BMI']
        Age = request.form['Age']
        prediction = loaded_model.predict([[Glucose,BMI,Age]])
        
        if prediction[0]==0:
            pred = "No Diabetes"
        else:
            pred = "Alert! You have diabetes."
        return render_template('index.html',pred=pred)
    else:
        return render_template('index.html')
    
       
  
if __name__ == "__main__":
    app.run(debug=True)

