from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the model
with open('rf_model.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        data = [float(request.form[field]) for field in ['GrLivArea', 'Bedrooms', 'TotalBath', 'KitchenAbvGr', 'Condition']]
        # Reshape and predict
        prediction = model.predict(np.array(data).reshape(1, -1))[0]
        return render_template('result.html', prediction=round(prediction, 2))
    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
