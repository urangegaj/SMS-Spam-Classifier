import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sb

data = pd.read_csv("SMSSpamCollection", sep="\t", names=["label", "message"])

print("First rows of dataset:")
print(data.head(), "\n")

print("Class counts:")
print(data["label"].value_counts(), "\n")

X_train, X_test, y_train, y_test = train_test_split(
    data["message"], data["label"], test_size=0.2, random_state=1
)

vectorizer = TfidfVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

clf = LogisticRegression(max_iter=200)
clf.fit(X_train_vec, y_train)

y_pred = clf.predict(X_test_vec)
print("Classification report:\n")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=["ham", "spam"])
sb.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
plt.title("Confusion Matrix")
plt.show()

def predict_message(msg):
    vec = vectorizer.transform([msg])
    return clf.predict(vec)[0]

print("\nSome test predictions:")
examples = ["You won a free iPhone!!!", 
            "Where should we go for lunch?", 
            "Congratulations you got selected!! click here"]
for e in examples:
    print(f"{e} --> {predict_message(e)}")
