import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import gradio as gr

# Load the dataset
data = pd.read_csv("dataset/Language Dataset.csv")

# Split the dataset into features and labels
X = data['Text']
y = data['language']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Text vectorization using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train a Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Evaluate the model
predictions = model.predict(X_test_vec)
print(classification_report(y_test, predictions))

# Define a prediction function
def predict_language(text):
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]
    return prediction

# Create a Gradio interface
interface = gr.Interface(
    fn=predict_language,
    inputs=gr.Textbox(lines=5, placeholder="Enter text here..."),
    outputs="label",
    title="Language Detection Model",
    description="Enter a text snippet to detect its language."
)

# Launch the interface
interface.launch()
