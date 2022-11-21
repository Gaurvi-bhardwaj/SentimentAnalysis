import os
from flask import Flask, render_template, request
import pickle

# Load the Naive Bayes model and TfidfVectorizer object from disk
filename = r"C:\Users\gourv\Downloads\Movies_Review_Classification_NLP-master\Movies_Review_Classification_NLP-master\Movies_Review_Classification.pkl"
classifier = pickle.load(open(filename, 'rb'))
cv = pickle.load(open(r"C:\Users\gourv\Downloads\Movies_Review_Classification_NLP-master\Movies_Review_Classification_NLP-master\count-Vectorizer.pkl",'rb'))
app = Flask(__name__)
app.env = "development"
app.run(host="localhost")
@app.route('/')
def home():
	return render_template('Movies Reviews Classifier1.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
    	message = request.form['message']
    	data = [message]
    	vect = cv.transform(data).toarray()
    	my_prediction = classifier.predict(vect)
    	return render_template('Movies Reviews Classifier.html', prediction=my_prediction)

if __name__ == '__main__':
    os.environ.setdefault('FLASK_ENV', 'development')
    app.run(debug=True)