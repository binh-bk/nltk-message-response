# nltk-message-response
NLTK to classify response form a text message with a visualization on a Flask app

# Quick start
- Required: `Python 3.6` or higher
- Set up virtual environment
`python3 -m venv venv`
- Activate virtual environment
`source venv/bin/activate`
- Install requirements 
`pip install - requirements.txt`
(You may have to update pip by `pip install pip -U`)

# Start Flask App
- Start the app by: `python3 run.py`
- This should appear on the terminal:
```
python3 run.py 
[nltk_data] Downloading package stopwords to /home/user/nltk_data...
[nltk_data]   Package stopwords is already up-to-date!
 * Serving Flask app "run" (lazy loading)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: on
 * Running on http://0.0.0.0:3001/ (Press CTRL+C to quit)
 * Restarting with stat
[nltk_data] Downloading package stopwords to /home/user/nltk_data...
[nltk_data]   Package stopwords is already up-to-date!
 * Debugger is active!
 * Debugger PIN: 952-012-911
```
- Main page of the Flask app
<img src='/img/main_page.png'>

- Result after querying a text
<img src='/img/result.png'>

# Machine Learning 
- this model use `nltk` and `scikit-learn` as the core component
- using `GridSearchCV` feature can be time-consuming. For a simple test run, use `build_model_simple()`, line 192 in `models/train_classifier.py` 
- results of training with `GridSearchCV` finetuned:
<img src='/models/evaluate_score.png'

# Going further:
- Check out `ETL Pipeline Preparation.ipynb` for data Extract-Transform-loading.
- and `ML Pipeline Preparation.ipynb` for train and testing machine learning model.

# Credits
- Udacity.come has prepared a framework for thsi project
- [RealPython.com](https://realpython.com/flask-by-example-part-3-text-processing-with-requests-beautifulsoup-nltk/) has a nice tutorial on building an Flask App with Nltk
