pip install scikit-learn

import streamlit as st
import pandas as pd
import numpy as np
from zipfile import ZipFile
import io
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# 1. Loading the SMS Spam Dataset
@st.cache_data
def load_sms_spam_data():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
    response = requests.get(url)
    
    with ZipFile(io.BytesIO(response.content)) as z:
        with z.open('SMSSpamCollection') as f:
            df = pd.read_csv(f, sep='\t', header=None, names=['label', 'message'])
    return df

# 2. Preprocessing Text Data (TF-IDF or Count Vectorizer)
def preprocess_text(data, method='tfidf'):
    if method == 'tfidf':
        vectorizer = TfidfVectorizer(stop_words='english')
    else:
        vectorizer = CountVectorizer(stop_words='english')
    
    X = vectorizer.fit_transform(data['message'])
    y = data['label'].map({'ham': 0, 'spam': 1})
    return X, y, vectorizer

# 3. Sidebar for Hyperparameter Selection with Unique Keys
def hyperparameter_selection(classifier_name):
    params = {}
    
    if classifier_name == "RandomForest":
        n_estimators = st.sidebar.slider("n_estimators", 50, 200, 100, key=f"{classifier_name}_n_estimators")
        max_depth = st.sidebar.slider("max_depth", 1, 20, 10, key=f"{classifier_name}_max_depth")
        min_samples_split = st.sidebar.slider("min_samples_split", 2, 10, 2, key=f"{classifier_name}_min_samples_split")
        params = {'n_estimators': n_estimators, 'max_depth': max_depth, 'min_samples_split': min_samples_split}
        
    elif classifier_name == "GradientBoosting":
        n_estimators = st.sidebar.slider("n_estimators", 50, 200, 100, key=f"{classifier_name}_n_estimators")
        learning_rate = st.sidebar.slider("learning_rate", 0.01, 0.5, 0.1, key=f"{classifier_name}_learning_rate")
        max_depth = st.sidebar.slider("max_depth", 1, 10, 3, key=f"{classifier_name}_max_depth")
        params = {'n_estimators': n_estimators, 'learning_rate': learning_rate, 'max_depth': max_depth}
        
    elif classifier_name == "LogisticRegression":
        C = st.sidebar.slider("C", 0.01, 1.0, 0.1, key=f"{classifier_name}_C")
        max_iter = st.sidebar.slider("max_iter", 100, 500, 200, key=f"{classifier_name}_max_iter")
        params = {'C': C, 'max_iter': max_iter}
        
    elif classifier_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 1.0, 0.1, key=f"{classifier_name}_C")
        kernel = st.sidebar.selectbox("kernel", ["linear", "rbf"], key=f"{classifier_name}_kernel")
        params = {'C': C, 'kernel': kernel}
        
    elif classifier_name == "KNN":
        n_neighbors = st.sidebar.slider("n_neighbors", 3, 10, 5, key=f"{classifier_name}_n_neighbors")
        weights = st.sidebar.selectbox("weights", ["uniform", "distance"], key=f"{classifier_name}_weights")
        params = {'n_neighbors': n_neighbors, 'weights': weights}
        
    elif classifier_name == "AdaBoost":
        n_estimators = st.sidebar.slider("n_estimators", 50, 200, 100, key=f"{classifier_name}_n_estimators")
        learning_rate = st.sidebar.slider("learning_rate", 0.01, 1.0, 0.1, key=f"{classifier_name}_learning_rate")
        params = {'n_estimators': n_estimators, 'learning_rate': learning_rate}
    
    return params

# 4. Training the Classifiers
def train_models(X_train, y_train, classifiers):
    models = []
    
    for name, clf in classifiers.items():
        params = hyperparameter_selection(name)
        st.write(f"Training {name} with params: {params}")
        clf.set_params(**params)
        clf.fit(X_train, y_train)
        models.append((name, clf))
    
    return models

# 5. Display Accuracy of Each Model
def display_accuracy(models, X_test, y_test):
    st.subheader("Model Accuracies")
    accuracies = []
    for name, model in models:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"{name}: {accuracy:.2f}")
        accuracies.append((name, accuracy))
    return accuracies

# 6. Plotting the Accuracy Chart
def plot_accuracy_chart(accuracies):
    names, values = zip(*accuracies)
    fig, ax = plt.subplots()
    ax.barh(names, values, color='skyblue')
    ax.set_xlabel("Accuracy")
    ax.set_title("Model Accuracy Comparison")
    st.pyplot(fig)

# 7. Confusion Matrix Plot
def plot_confusion_matrix(y_true, y_pred, classifier_name):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f"Confusion Matrix - {classifier_name}")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    st.pyplot(fig)

# 8. Class Distribution Plot
def plot_class_distribution(data):
    st.subheader("Class Distribution")
    class_counts = data['label'].value_counts()
    fig, ax = plt.subplots()
    class_counts.plot(kind='bar', color=['green', 'red'], ax=ax)
    ax.set_title("Class Distribution of Spam and Ham")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    st.pyplot(fig)

# 9. Word Frequency Plot for Spam and Ham
def plot_word_frequencies(data, vectorizer):
    st.subheader("Most Frequent Words in Spam and Ham")
    spam_words = ' '.join(data[data['label'] == 'spam']['message'])
    ham_words = ' '.join(data[data['label'] == 'ham']['message'])
    
    spam_counter = Counter(spam_words.split())
    ham_counter = Counter(ham_words.split())
    
    spam_common = pd.DataFrame(spam_counter.most_common(10), columns=['word', 'count'])
    ham_common = pd.DataFrame(ham_counter.most_common(10), columns=['word', 'count'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    sns.barplot(x='count', y='word', data=spam_common, ax=ax1, palette='Reds_r')
    ax1.set_title("Most Frequent Words in Spam")
    
    sns.barplot(x='count', y='word', data=ham_common, ax=ax2, palette='Greens_r')
    ax2.set_title("Most Frequent Words in Ham")
    
    st.pyplot(fig)

# 10. Main Function to Run the App
def main():
    st.title("SMS Spam Detection with Ensemble Learning [Ajay Kumar Jha]")

    # Load dataset
    data = load_sms_spam_data()
    st.write("Dataset Preview:")
    st.dataframe(data.head())

    # Show class distribution
    plot_class_distribution(data)

    # Sidebar for user input
    vectorizer_method = st.sidebar.selectbox("Vectorizer Method", ["tfidf", "count"])
    test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2, key="test_size")
    classifiers_to_use = st.sidebar.multiselect(
        "Select Classifiers",
        ["RandomForest", "GradientBoosting", "LogisticRegression", "SVM", "KNN", "AdaBoost"],
        ["RandomForest", "GradientBoosting", "LogisticRegression"]
    )

    # Preprocess data
    X, y, vectorizer = preprocess_text(data, vectorizer_method)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Initialize selected classifiers
    classifiers = {
        "RandomForest": RandomForestClassifier(),
        "GradientBoosting": GradientBoostingClassifier(),
        "LogisticRegression": LogisticRegression(),
        "SVM": SVC(),
        "KNN": KNeighborsClassifier(),
        "AdaBoost": AdaBoostClassifier()
    }
    
    # Filter selected classifiers
    selected_classifiers = {name: classifiers[name] for name in classifiers_to_use}

    # Train the models
    models = train_models(X_train, y_train, selected_classifiers)
    
    # Display accuracies
    accuracies = display_accuracy(models, X_test, y_test)
    
    # Plot accuracy comparison
    plot_accuracy_chart(accuracies)
    
    # Show confusion matrix for each model
    for name, model in models:
        y_pred = model.predict(X_test)
        plot_confusion_matrix(y_test, y_pred, name)

    # Show word frequencies
    plot_word_frequencies(data, vectorizer)

# Run the app
if __name__ == '__main__':
    main()
