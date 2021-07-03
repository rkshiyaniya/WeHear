# import libraries
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from flask import jsonify
from joblib import load

# Initialize the flask App
app = Flask(__name__)

# Mappind Dictionary

tab_map = {
    "prc_500_mg": 0,
    "neurobion": 1,
    "dolo_650": 2,
    "supradyn": 3,
    "crocin": 4,
    "combiflame": 5,
    "prc_650_mg": 6,
    "aspirin_75_mg": 7,
    "danp": 8,
    "ltk_h": 9
}

op_map = {value: key for key, value in tab_map.items()}

# model = pickle.load(open('final_model.pkl', 'rb'))
model = load('model.pkl')


# function for prediction
def make_pred(arr):
    preds = model.predict_proba(arr.reshape(1, -1))
    return op_map.get(preds.argmax())


# default page of our web-app
@app.route('/')
def home():
    return render_template('index.html')


# default page of our web-app
@app.route('/predict', methods=['POST'])
def prediction():
    arr = request.get_json()
    arr = np.array(arr['arr'])
    preds = model.predict_proba(arr.reshape(1, -1))
    acc = np.amax(preds)
    output = op_map.get(preds.argmax())
    return jsonify(med_name=output, acc=str(acc*100))


if __name__ == "__main__":
    app.run(debug=True)
