# phishing_detector.py

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = os.path.join("data", "training_data.csv")


def load_data(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Training data file not found: {file_path}. "
            "Make sure data/training_data.csv exists."
        )

    df = pd.read_csv(file_path)

    required_columns = {"text", "label"}
    if not required_columns.issubset(df.columns):
        raise ValueError("CSV file must contain 'text' and 'label' columns.")

    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str).str.lower().str.strip()

    valid_labels = {"phishing", "legitimate"}
    df = df[df["label"].isin(valid_labels)]

    if df.empty:
        raise ValueError("No valid training data found after cleaning.")

    return df


def train_model(df: pd.DataFrame):
    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
        ("classifier", MultinomialNB())
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return model, accuracy, report


def predict_email(model, email_text: str):
    probabilities = model.predict_proba([email_text])[0]
    classes = model.classes_

    class_probs = dict(zip(classes, probabilities))
    prediction = model.predict([email_text])[0]
    confidence = class_probs[prediction] * 100

    return prediction, confidence, class_probs


def show_tips(prediction: str):
    print("\nSecurity Tips")
    print("-------------")

    if prediction == "phishing":
        print("- Do not click suspicious links.")
        print("- Do not share passwords or OTPs.")
        print("- Verify the sender address carefully.")
        print("- Contact the company directly through its official website.")
    else:
        print("- This email looks legitimate based on the model.")
        print("- Still check attachments and links before taking action.")
        print("- Be careful with urgent requests asking for sensitive information.")


def main():
    print("Phishing Email Detector (AI)")
    print("-" * 28)

    try:
        df = load_data(DATA_PATH)
        model, accuracy, report = train_model(df)
    except Exception as error:
        print(f"Error: {error}")
        return

    print(f"Model trained successfully on {len(df)} emails.")
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")

    print("\nEnter the email text to analyze:")
    email_text = input("> ").strip()

    if not email_text:
        print("No email text entered.")
        return

    prediction, confidence, class_probs = predict_email(model, email_text)

    print("\nPrediction Result")
    print("-" * 18)
    print(f"Prediction: {prediction.capitalize()}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"Phishing Probability: {class_probs.get('phishing', 0) * 100:.2f}%")
    print(f"Legitimate Probability: {class_probs.get('legitimate', 0) * 100:.2f}%")

    show_tips(prediction)

    print("\nModel Report")
    print("-" * 12)
    print(report)


if __name__ == "__main__":
    main()
