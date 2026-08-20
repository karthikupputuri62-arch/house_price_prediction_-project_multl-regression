from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
with open("house.pkl", "rb") as file:
    model = pickle.load(file)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        # ==========================================
        # HOUSE INFORMATION
        # ==========================================

        bedrooms = float(request.form["bedrooms"])
        bathrooms = float(request.form["bathrooms"])
        sqft_living = float(request.form["sqft_living"])
        sqft_lot = float(request.form["sqft_lot"])
        floors = float(request.form["floors"])
        waterfront = float(request.form["waterfront"])
        view = float(request.form["view"])
        condition = float(request.form["condition"])
        sqft_above = float(request.form["sqft_above"])
        sqft_basement = float(request.form["sqft_basement"])
        yr_built = float(request.form["yr_built"])
        yr_renovated = float(request.form["yr_renovated"])

        # ==========================================
        # CITY
        # ==========================================

        city = float(request.form["city"])

        # ==========================================
        # DATE
        # ==========================================

        day = float(request.form["day"])
        month = float(request.form["month"])
        year = float(request.form["year"])

        # ==========================================
        # 17 FEATURES
        # ==========================================

        features = np.array([[
            bedrooms,
            bathrooms,
            sqft_living,
            sqft_lot,
            floors,
            waterfront,
            view,
            condition,
            sqft_above,
            sqft_basement,
            yr_built,
            yr_renovated,
            city,
            day,
            month,
            year,
            0
        ]])

        # Prediction
        prediction = model.predict(features)[0]

        return render_template(
            "index.html",
            prediction=round(prediction, 2)
        )

    except Exception as e:

        return render_template(
            "index.html",
            error=str(e)
        )


if __name__ == "__main__":
    app.run(debug=True)