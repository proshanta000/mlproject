# app.py (Your Flask application file)
from flask import Flask, request, render_template
import numpy as np
import pandas as pd

# Assuming these imports are correct and available in your environment
# from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipline import PredictPipeline, CustomData


application = Flask(__name__)

app = application

# Route for the home page (displays the input form)
@app.route('/')
def home():
    """
    Renders the main input form page (index.html).
    """
    return render_template('index.html')

# Route to handle form submission and display prediction
@app.route('/predict_datapoint', methods=['POST']) # Changed route name to match HTML form action
def predict_datapoint():
    """
    Handles the form submission, collects data, makes a prediction,
    and renders the result page.
    """
    # This route should only handle POST requests from the form
    if request.method == 'POST':
        # Create a CustomData object from form inputs
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))
        )
        
        # Convert CustomData to a DataFrame for the prediction pipeline
        pred_df = data.get_data_as_data_frame()
        print(pred_df) # For debugging: print the DataFrame being sent to prediction

        # Initialize and run your prediction pipeline
        predict_pipeline = PredictPipeline()
        result = predict_pipeline.predict(pred_df)
        
        # Render the result page, passing the prediction result
        # Ensure the variable name 'results' matches the placeholder in result.html
        return render_template('result.html', results=result[0])
    # No 'GET' method here for '/predict_datapoint', as the form is served by '/'

if __name__ == "__main__":
    # Ensure you have a 'templates' folder in the same directory as app.py
    # and that 'index.html' and 'result.html' are inside it.
    app.run(host='0.0.0.0', port=5000, debug=True)