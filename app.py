from flask import Flask, request, jsonify
from flasgger import Swagger, swag_from
from flask_cors import CORS   # <-- add this
import pickle
import pandas as pd
import numpy as np
import logging

app = Flask(__name__)
CORS(app)   # <-- enable CORS
swagger = Swagger(app)

# Logging setup
logging.basicConfig(level=logging.INFO)

# Load the saved model
with open("trainModel.pkl", "rb") as f:
    model = pickle.load(f)

# Define all columns expected by the model
expected_cols = ['authors', 'title', 'publication_date', 'ratings_count',
                 '  num_pages', 'text_reviews_count', 'language_code', 'publisher']

@app.route('/predict', methods=['POST'])
@swag_from({
    'tags': ['Prediction'],
    'parameters': [
        {
            "name": "body",
            "in": "body",
            "required": True,
            "schema": {
                "type": "object",
                "properties": {
                    "authors": {"type": "string"},
                    "title": {"type": "string"},
                    "publication_date": {"type": "string", "example": "2000-01-01"},
                    "ratings_count": {"type": "integer"},
                    "num_pages": {"type": "integer"},
                    "text_reviews_count": {"type": "integer"},
                    "language_code": {"type": "string"},
                    "publisher": {"type": "string"}
                },
                "required": ["authors", "title"]
            }
        }
    ],
    'responses': {
        200: {
            'description': 'Predicted average rating',
            'examples': {'average_rating': 4.2}
        }
    }
})
def predict():
    try:
        data = request.get_json()
        
        # Convert JSON to DataFrame
        df = pd.DataFrame([data])

        # Fill missing columns with defaults
        for col in expected_cols:
            if col not in df.columns:
                if col in ['ratings_count', '  num_pages', 'text_reviews_count']:
                    df[col] = 0
                else:
                    df[col] = ''

        # Ensure proper types
        df['publication_date'] = pd.to_datetime(df['publication_date'], errors='coerce')
        df['ratings_count'] = df['ratings_count'].astype(int)
        df['  num_pages'] = df['  num_pages'].astype(int)
        df['text_reviews_count'] = df['text_reviews_count'].astype(int)

        # Predict using the loaded pipeline
        prediction = model.predict(df) 

        return jsonify({'average_rating': float(prediction[0])})

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)



# http://localhost:5000/apidocs