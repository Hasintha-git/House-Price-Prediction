from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# Define global variable for the model
model = None

# Load the model from the .pkl file
try:
    with open('model/house_price_prediction_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully.")

except FileNotFoundError:
    print("Error: Model file not found.")
except EOFError:
    print("Error: Model file is empty or corrupted.")
except Exception as e:
    print("Error:", e)

# Load data from test.csv
test_data = pd.read_csv('data/test.csv')

# Get unique values for dropdown options
MSZoning_options = test_data['MSZoning'].unique().tolist()
Street_options = test_data['Street'].unique().tolist()
SaleCondition_options = test_data['SaleCondition'].unique().tolist()
BedroomAbvGr_options = test_data['BedroomAbvGr'].unique().tolist()
YearBuilt_options = test_data['YearBuilt'].unique().tolist()
HouseStyle_options = test_data['HouseStyle'].unique().tolist()

@app.route('/')
def home():
    return render_template('index.html', MSZoning_options=MSZoning_options, Street_options=Street_options, 
                           SaleCondition_options=SaleCondition_options, BedroomAbvGr_options=BedroomAbvGr_options, 
                           YearBuilt_options=YearBuilt_options, HouseStyle_options=HouseStyle_options)

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Not Found'}), 404

@app.route('/predict', methods=['POST'])
def predict():
    global model  # Access the global model variable
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'})

        # Parse JSON data from request body
        data = request.json

        # Validate request data
        if not data or not all(key in data for key in ('MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'SaleCondition', 'YearBuilt', 'BedroomAbvGr', 'HouseStyle')):
            return jsonify({'error': 'Invalid request data'}), 400

        MSSubClass = int(data['MSSubClass'])
        MSZoning = data['MSZoning']
        LotFrontage = float(data['LotFrontage'])
        LotArea = int(data['LotArea'])
        Street = data['Street']
        SaleCondition = data['SaleCondition']
        YearBuilt = int(data['YearBuilt'])
        BedroomAbvGr = int(data['BedroomAbvGr'])
        HouseStyle = data['HouseStyle']

        # Create DataFrame with user input
        X_test = pd.DataFrame({
            'MSSubClass': [MSSubClass],
            'MSZoning': [MSZoning],
            'LotFrontage': [LotFrontage],
            'LotArea': [LotArea],
            'Street': [Street],
            'SaleCondition': [SaleCondition],
            'YearBuilt': [YearBuilt],
            'BedroomAbvGr': [BedroomAbvGr],
            'HouseStyle': [HouseStyle],
        })

        print("X_test :", X_test)
        # Make predictions
        prediction = model.predict(X_test)

        decoded_prices = np.exp(prediction)
        print("prediction :", decoded_prices)
        return jsonify({'prediction': decoded_prices[0]})

    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except KeyError as ke:
        return jsonify({'error': f'Missing key: {ke}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
