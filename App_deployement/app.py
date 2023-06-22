import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = []

    for feature_name, feature_value in request.form.items():
        if feature_name == 'Revenu du Conjoint':
            try:
                float_value = float(feature_value)
                int_features.append(float_value)
            except ValueError:
                error_message = f"Valeur incorrecte pour {feature_name}"
                return render_template('index.html', prediction_text=error_message)
        else:
            if feature_value.lower() == 'oui':
                int_features.append(1)
            elif feature_value.lower() == 'non':
                int_features.append(0)
            else:
                error_message = f"Valeur incorrecte pour {feature_name}"
                return render_template('index.html', prediction_text=error_message)

    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0])

    if output == 1:
        prediction_text = "Vous êtes éligible à ouvrir un crédit"
    else:
        prediction_text = "Vous n'êtes pas éligible à ouvrir un crédit"

    return render_template('index.html', prediction_text=prediction_text)


@app.route('/predict_api', methods=['POST'])
def predict_api():
    '''
    For direct API calls through request
    '''
    data = request.get_json(force=True)
    int_features = []

    for feature_name, feature_value in data.items():
        if feature_name == 'Revenu du Conjoint':
            try:
                float_value = float(feature_value)
                int_features.append(float_value)
            except ValueError:
                error_message = f"Valeur incorrecte pour {feature_name}"
                return jsonify(error=error_message)
        else:
            if feature_value.lower() == 'oui':
                int_features.append(1)
            elif feature_value.lower() == 'non':
                int_features.append(0)
            else:
                error_message = f"Valeur incorrecte pour {feature_name}"
                return jsonify(error=error_message)

    prediction = model.predict([np.array(int_features)])

    output = prediction[0]
    return jsonify(output)


if __name__ == "__main__":
    app.run(debug=True)
