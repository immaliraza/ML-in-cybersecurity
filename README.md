# Email Spam Classifier

This project implements an **Email Spam Classifier** using a **Decision Tree Classifier**. The classifier processes email data from the `spamassassin-public-corpus` dataset to distinguish between spam and ham (non-spam) emails. The feature extraction is done using **HashingVectorizer** and **TfidfTransformer**, and model performance is evaluated using **accuracy score** and **confusion matrix**.



## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Code Walkthrough](#code-walkthrough)
- [Results](#results)
- [Improvements](#improvements)



## Project Overview
This project trains a **Decision Tree Classifier** to classify email messages as spam or ham. The main steps include:
1. Loading the dataset from the `spamassassin-public-corpus` folder.
2. Preprocessing and feature extraction using **HashingVectorizer** and **TfidfTransformer**.
3. Training a **Decision Tree Classifier** to classify emails.
4. Evaluating the classifier using accuracy score and confusion matrix.



## Dataset
The dataset used for training and testing is the **SpamAssassin Public Corpus**, a popular open-source dataset for spam email classification. It contains two directories:
- **spam**: Contains spam emails.
- **ham**: Contains non-spam (ham) emails.

Make sure the dataset is organized as follows:
```
./spamassassin-public-corpus/
    ├── spam/
    │     └── [list of spam email files]
    └── ham/
          └── [list of ham email files]
```



## Prerequisites

To run the project, you need to have the following Python libraries installed:
- `scikit-learn`
- `os`

Install the necessary libraries with:
```bash
pip install scikit-learn
```



## Project Structure
```
📦 Project Root
 ┣ 📂 spamassassin-public-corpus
 ┃ ┣ 📂 spam
 ┃ ┃ ┗ 📜 [spam email files]
 ┃ ┗ 📂 ham
 ┃   ┗ 📜 [ham email files]
 ┗ 📜 email_spam_classifier.py
```



## How It Works
1. **Data Loading**: Loads emails from the "spam" and "ham" folders.
2. **Labeling**: Assigns a label of `0` for spam and `1` for ham.
3. **Feature Extraction**: Converts email text into numerical features using **HashingVectorizer** and **TfidfTransformer**.
4. **Training**: Trains a **Decision Tree Classifier** on 80% of the data.
5. **Evaluation**: Tests the classifier on 20% of the data and outputs the accuracy and confusion matrix.


## Code Walkthrough

```python
# Import necessary libraries
from sklearn.feature_extraction.text import TfidfTransformer, HashingVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import os
```
**Imports**: The required libraries are imported for file processing, feature extraction, model training, and evaluation.


```python
# Step 1: Load the dataset from the "spamassassin-public-corpus" folder
def load_emails_from_folder(folder_path):
    emails = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r', encoding='latin1') as file:
            emails.append(file.read())
    return emails
```
**Data Loading**: This function reads all files from the given folder path and returns the content of each email as a list of strings.



```python
# Paths to spam and ham datasets
spam_path = "./spamassassin-public-corpus/spam"
ham_path = "./spamassassin-public-corpus/ham"

# Step 2: Load emails and assign labels
spam_emails = load_emails_from_folder(spam_path)
ham_emails = load_emails_from_folder(ham_path)
```
**Load Dataset**: Loads spam and ham emails from the respective folders and stores them in separate lists.



```python
# Step 3: Create lists for email contents and labels
email_contents = spam_emails + ham_emails
email_labels = [0] * len(spam_emails) + [1] * len(ham_emails)
```
**Data Labeling**: Combines spam and ham email content into one list and creates corresponding labels (`0` for spam, `1` for ham).



```python
# Step 4: Perform train-test split (80%-20%)
X_train, X_test, y_train, y_test = train_test_split(
    email_contents, email_labels, test_size=0.2, random_state=42)
```
**Train-Test Split**: Splits the data into training (80%) and testing (20%) sets.



```python
# Step 5: Train a DecisionTreeClassifier with HashingVectorizer and TfidfTransformer
vectorizer = HashingVectorizer(n_features=2**12)  # Use a large number of features for sparse text
tfidf_transformer = TfidfTransformer()
classifier = DecisionTreeClassifier()
```
**Feature Extraction**: Converts email text into numerical features using **HashingVectorizer** and **TfidfTransformer**.


```python
# Transform data
X_train_vec = tfidf_transformer.fit_transform(vectorizer.transform(X_train))
X_test_vec = tfidf_transformer.transform(vectorizer.transform(X_test))

# Train the classifier
classifier.fit(X_train_vec, y_train)
```
**Model Training**: Trains the **Decision Tree Classifier** on the training data.



```python
# Step 6: Evaluate the model
y_pred = classifier.predict(X_test_vec)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Print results
print("Model Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
```
**Evaluation**: Predicts on the test set, calculates accuracy, and displays the confusion matrix.



## Results
When you run the script, you will see the following output:
```
Model Accuracy: [example: 0.85]
Confusion Matrix:
 [[TP  FP]
  [FN  TN]]
```
- **Accuracy**: Measures how well the classifier predicts on test data.
- **Confusion Matrix**: Provides insight into true positives, true negatives, false positives, and false negatives.



## Improvements
Here are some ways to improve the classifier:
- **Try different classifiers**: Test models like Random Forest or Naive Bayes.
- **Use better text preprocessing**: Remove stopwords, lemmatize, or stem text before vectorization.
- **Tune hyperparameters**: Optimize the hyperparameters of **DecisionTreeClassifier**.
- **Use more features**: Use **TF-IDF Vectorizer** instead of **HashingVectorizer** for better feature representation.


Check out the full article for a detailed explanation of the project and the steps involved on Medium: https://medium.com/@mraza1/introduction-52b3d6598d38


