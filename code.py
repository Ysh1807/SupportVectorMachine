import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
def load_data():
    # Example: Load IMDB movie reviews dataset
    data = pd.read_csv("IMDB Dataset.csv")  # Adjust path
    data['label'] = data['sentiment'].map({'positive': 1, 'negative': 0})
    return data['review'], data['label']

# Preprocess text
def preprocess_text(text):
    import re
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    text = re.sub(r'[^\w\s]', '', text.lower())  # Remove punctuation and lowercase
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

# Main
if __name__ == "__main__":
    # Load and preprocess data
    reviews, labels = load_data()
    reviews = reviews.apply(preprocess_text)

    # Convert text to numerical features using TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(reviews).toarray()
    y = labels

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train SVM classifier
    svm = SVC(kernel='linear', C=1.0)  # Linear kernel for text data
    svm.fit(X_train, y_train)

    # Evaluate the model
    y_pred = svm.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
