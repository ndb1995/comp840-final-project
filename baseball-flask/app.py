from flask import Flask, render_template, request, redirect, jsonify
import os
import numpy as np
from sklearn.externals import joblib
import pickle
app = Flask(__name__)

def unpickle(path):
	my_path = path
	model = open(my_path, 'rb')
	unpickeled = pickle.load(model)
	return unpickeled


# default page
@app.route("/", methods=['GET'])
def index():
    return render_template("index.html")
	
# prediction page	
@app.route("/results", methods=['POST'])
def predictions():
	text = request.form['text']
	text = text.upper()
	
	
	grid_search = joblib.load('../grid_search.pkl')
	#random_forest = joblib.load('../random_forest.pkl')
	sgd_regressor = joblib.load('../sgd_regressor.pkl')

	if(text == "GRID SEARCH"):
		grid_search_results = grid_search.predict(unpickle("../grid_search_X_test.pkl"))
		return render_template("results.html", grid_search_results)
	elif(text == "SGD REGRESSION"):
		sgd_regressor_results = grid_search.predict(unpickle("../sgd_reg_X_test.pkl"))
		return render_template("results.html", sgd_regressor_results)
	else:
		text = "Not an algorithm that can be processed!"
		return render_template("results.html", text = text)
		
if __name__ == "__main__":
    app.run(debug=False, port=5000, host='0.0.0.0') 
	
	
