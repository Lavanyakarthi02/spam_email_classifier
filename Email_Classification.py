import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Download stopwords
nltk.download('stopwords')

# Read Dataset
try:
    df = pd.read_csv(
        "Spam.csv",
        sep="\t",
        names=["label", "message"],
        encoding="latin-1"
    )
except Exception as e:
    print("Error reading Spam.csv:", e)
    exit()

# Remove empty rows
df.dropna(inplace=True)

# Clean labels
df["label"] = df["label"].astype(str).str.strip().str.lower()

# Keep only ham and spam
df = df[df["label"].isin(["ham", "spam"])]

# Convert labels to numbers
df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

print("\nDataset Shape:", df.shape)
print(df.head())

print("\nLabel Count:")
print(df["label"].value_counts())

# Text preprocessing
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = []
    for word in text.split():
        word = word.lower()
        if word not in stop_words:
            words.append(word)
    return " ".join(words)
df["message"] = df["message"].apply(clean_text)

# Feature Extraction
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(df["message"])
y = df["label"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

#==========================
#Naive Bayes
#==========================
nb = MultinomialNB()
nb.fit(X_train, y_train)
nb_pred = nb.predict(X_test)
print("\n NAIVE BAYES ")
print("Accuracy:", accuracy_score(y_test, nb_pred))
print(classification_report(y_test, nb_pred))

# ==========================
# SVM
# ==========================
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)
print("\n SVM ")
print("Accuracy:", accuracy_score(y_test, svm_pred))
print(classification_report(y_test, svm_pred))