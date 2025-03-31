from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import json

app = Flask(__name__)

@app.route('/')
def index():
    total_sqft, location, bhk, bath, msg = None, None, None, None, None
    return render_template('index.html', total_sqft=total_sqft, location=location, bhk=bhk, bath=bath, msg =  msg)

@app.route('/predict_home_price', methods=['POST'])
def predict_home_price():

    with open("./artifacts/columns.json", "r") as f:
        __data_columns = json.load(f)['data_columns']

    with open('./artifacts/banglore_home_prices_model.pickle', 'rb') as f:
        __model = pickle.load(f)

    total_sqft = float(request.form['total_sqft'])
    location = request.form['location']
    bhk = int(request.form['bhk'])
    bath = int(request.form['bath'])

    if location not in __data_columns:
        location = 'other'

    location_id = np.where(np.array(__data_columns[3:]) == location, 1, 0)
    x = np.zeros(len(__data_columns))
    x[0] = total_sqft
    x[1] = bath
    x[2] = bhk
    x[3:] = location_id
    msg = round(__model.predict([x])[0], 3)
    return render_template('index.html', total_sqft=total_sqft, location=location, bhk=bhk, bath=bath, msg =  msg)

if __name__ == "__main__":
    print("Starting Python Flask Server For Home Price Prediction...")
    app.run(debug = True)