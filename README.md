
# Customer Experience Sentiment Classifier

This project builds a sentiment classification model to analyze customer feedback and determine whether comments are **positive** or **negative**. It leverages a machine learning pipeline using **TF-IDF vectorization** and a **Linear Support Vector Classifier (LinearSVC)** to perform sentiment analysis on customer experience data.

## 📊 Dataset

The dataset consists of customer comments labeled as `pos` (positive) or `neg` (negative) sentiments. It is loaded from a CSV file named `customer_comments_data.csv`.

Key statistics:
- Number of samples: *[add actual number if known]*
- Features: `comment` (text), `label` (target)

## 🏗️ Model Pipeline

The model pipeline includes:

1. **Text Vectorization:** using `TfidfVectorizer()` to convert text into numerical feature vectors.
2. **Stopwords Handling:**
   - Initially built without stopwords removal.
   - Extended with combined **English stopwords** and **custom stopwords** for improved preprocessing.
3. **Classification Model:** using `LinearSVC()` for supervised text classification.
4. **Model Evaluation:**
   - Accuracy on training and testing sets
   - Confusion matrix
   - Classification report (precision, recall, F1-score)

## 🚀 Key Steps

- Data loading and inspection
- Data cleaning and missing value checks
- Feature-target split
- Train-test split (70% train / 30% test)
- Model training
- Model evaluation
- Experimentation with standard and custom stopwords

## 📝 Usage

To run the notebook:

1. Install required packages:
   ```bash
   pip install pandas numpy scikit-learn
   ```
2. Run `customer_experience_sentiment_classification.ipynb` in Jupyter Notebook or JupyterLab.


## 🔬 Future Improvements

- Hyperparameter tuning for `LinearSVC`
- Integration of additional NLP preprocessing (lemmatization, n-grams)
- Model comparison with other classifiers (e.g., Logistic Regression, Naive Bayes)


