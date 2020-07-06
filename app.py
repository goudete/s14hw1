from flask import Flask, render_template
import joblib

app = Flask(__name__)

@app.route('/')
def index():
    regr_model = joblib.load('regr_model.pkl')
    tree_model = joblib.load('tree_regressor.pkl')
    # Make prediction - features = ['BEDS', 'BATHS', 'SQFT', 'AGE', 'LOTSIZE', 'GARAGE']
    regr_prediction = regr_model.predict([[4, 2.5, 3005, 15, 17903.0, 1]])[0][0].round(1)
    tree_prediction = tree_model.predict([[4, 2.5, 3005, 15, 17903.0, 1]])

    regr_prediction = str(regr_prediction)
    return render_template('index.html', regr_model=regr_prediction, tree_model=tree_prediction)
