import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from flask import Flask
from flask import request
from flask import jsonify


model_file = 'model1.bin'
dv_file = 'dv.bin'

with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)

with open(dv_file, 'rb') as f_in:
    dv = pickle.load(f_in)

app = Flask('CreditScoring')

# customer = {"job": "retired", "duration": 445, "poutcome": "success"}

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0,1]

    result = {
        'Probability': float(y_pred)
    }

    return jsonify(result)

    # print("input",customer)
    # print("probability", y_pred)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9697)


