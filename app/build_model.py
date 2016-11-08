import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
try:
    from sklearn.model_selection import train_test_split
except:
    from sklearn.cross_validation import train_test_split
    
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline, FeatureUnion, make_union, make_pipeline


df = pd.read_csv("data/smsspamcollection/SMSSpamCollection", sep="\t", 
                 header=None, 
                 names=["target", "text"])

X = df["text"]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Using Make Pipeline and Make Union
log_reg_model = make_pipeline(CountVectorizer(), LogisticRegression())

log_reg_model.fit(X_train, y_train)

joblib.dump(log_reg_model, "models/spam_ham.pkl")
