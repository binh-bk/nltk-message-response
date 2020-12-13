# nltk-message-response
NLTK to classify response form a text message with a visualization on a Flask app

# Quick start
- Required: `Python 3.6` or higher
- Set up virtual environment
`python3 -m venv venv`
- Activate virtual environment
`source venv/bin/activate`
- Install requirements 
`pip install -r requirements.txt`
(You may have to update `pip` by `pip install pip -U`)

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

# Data Extract-Transform-Loading (ETL)
- <em>inside `data` folder</em>, run `process_data.py` to clean, combine data like this:
```python3 process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db```
- Data is saved to `message_response` table by default. Look for line 66 in the script to customize.

# Machine Learning 
- this model uses `nltk` and `scikit-learn` libraries as the core components
- using `GridSearchCV` feature can be time-consuming. For a simple test run, use `build_model_simple()`, line 192 in `models/train_classifier.py` 
- run the script, <em>inside `models` folder</em>, enter this to the terminal:
```python3 train_classifier.py ../data/DisasterResponse.db classifier.pkl```
- for a simple model, training took about 1 minute, and evaluation on test data took about 2 minutes. A summary of evaluation is saved to `evaluation_score.txt` and `test_score.png`
- results of training with `GridSearchCV` fine-tuning:
<img src='/models/evaluate_score_tuned.png'>

# Going further:
- Check out `ETL Pipeline Preparation.ipynb` for data Extract-Transform-loading.
- and `ML Pipeline Preparation.ipynb` for train and testing machine learning model.

# Credits
- Udacity.com has prepared a framework for this project
- [RealPython.com](https://realpython.com/flask-by-example-part-3-text-processing-with-requests-beautifulsoup-nltk/) has a nice tutorial on building an Flask App with Nltk
