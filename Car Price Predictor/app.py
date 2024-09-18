from flask import Flask, render_template, request, redirect
from flask_cors import CORS, cross_origin
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
df_car = pd.read_csv("./Cleaned_car_data.csv")
model = pickle.load(open("Model.pkl", "rb"))


@app.route('/')
def index():
    compaines = sorted(df_car["company"].unique())
    car_models = sorted(df_car["name"].unique())
    year = sorted(df_car["year"].unique(), reverse=True)
    fuel_type = df_car["fuel_type"].unique()

    return render_template("index.html", companies=compaines, car_models=car_models, years=year, fuel_types=fuel_type)


@app.route("/predict", methods=["post"])
def predict():
    company = request.form.get("company")
    car_model = request.form.get("car_models")
    year = request.form.get("year")
    fuel_type = request.form.get("fuel")
    driven = request.form.get("kilo_driven")

    prediction = model.predict(pd.DataFrame([[car_model, company, year, driven, fuel_type]], columns=[
                               "name", "company", "year", "kms_driven", "fuel_type"]))

    return str(np.round(prediction[0], 2))
    # return render_template("index.html", prediction=prediction)
    # return str(prediction[0]) + " " + str(prediction[1])
    # return jsonify({'Prediction':str(prediction)})


if __name__ == "__main__":
    app.run(debug=True)
