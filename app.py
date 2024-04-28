from flask import Flask, render_template, request
import joblib

# Load the trained pipeline
pipeline = joblib.load('trained_pipeline.pkl')  # Replace 'trained_pipeline.pkl' with the path to your trained pipeline file

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    content = request.form['content']
    prediction = predict_extremism(content)
    return render_template('result.html', prediction=prediction)

def predict_extremism(content):
    # Predict whether the content is extremist or not
    prediction = pipeline.predict([content])[0]
    return 'Extremist' if prediction == 1 else 'Not Extremist'

if __name__ == '__main__':
    app.run(debug=True)
