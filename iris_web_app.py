import joblib
from flask import Flask, request, jsonify, render_template
import numpy as np

app = Flask(__name__)

# Load the model
print('LOADING THE MODEL')
model = joblib.load('model/iris_classifier.joblib')

# Label dictionary
label_mapping = {0:'setosa', 1:'versicolor', 2:'virginica'}

@app.route('/home')
def home():
    return render_template('index.html')

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.json
        
        # Assuming data contains features for prediction
        features = np.array(data['features'])
        
        # Convert features to the format expected by XGBoost (if necessary)
        # Make prediction
        print('MAKING PREDICTIONS')
        prediction = model.predict([features]).tolist()[0]

        # Mapping the label name
        prediction = label_mapping[prediction]

        # Prepare JSON response
        print('PREPARING THE RESPONSE')
        response = {'prediction': prediction}  # Convert numpy array to list
        
        print('SENDING THE RESPONSE')
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)})

# Run the app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)