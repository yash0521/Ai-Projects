from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)
data = pd.read_csv("Cleaned_data.csv")
pipe = pickle.load(open("RidgeModel.pkl", "rb"))


@app.route('/')
def index():

    locations = sorted(data["location"].unique())
    return render_template('index.html', locations=locations)


@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk = request.form.get('bhk')
    bath = request.form.get('bathrooms')
    total_sqft = request.form.get('total_sqft')
    print(location, bhk, bath, total_sqft)

    input = pd.DataFrame(
        [[location, total_sqft, bath, bhk]], columns=["location", "total_sqft", "bath", "bhk"])
    output = pipe.predict(input)[0] * 1e5  # same as * 100000

    return str(np.round(output, 2))


if __name__ == "__main__":
    app.run(debug=True, port=5001)
