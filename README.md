
# IMDB Movie Review Sentiment Analysis using Simple RNN

An end-to-end Deep Learning project for classifying movie reviews as **Positive** or **Negative** using Word Embeddings and a Simple Recurrent Neural Network (RNN).

The project uses **TensorFlow/Keras** for deep learning and **Streamlit** to provide an interactive web application where users can enter movie reviews and receive sentiment predictions.

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?logo=keras&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-Sentiment%20Analysis-blue)
![License](https://img.shields.io/badge/License-GPLv3-blue)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Key Features](#key-features)
4. [Technology Stack](#technology-stack)
5. [Dataset](#dataset)
6. [System Architecture](#system-architecture)
7. [Project Workflow](#project-workflow)
8. [Model Architecture](#model-architecture)
9. [Project Structure](#project-structure)
10. [Installation and Setup](#installation-and-setup)
11. [Running the Application](#running-the-application)
12. [Prediction Process](#prediction-process)
13. [Model Evaluation](#model-evaluation)
14. [Learning Outcomes](#learning-outcomes)
15. [Future Improvements](#future-improvements)
16. [License](#license)
17. [Author](#author)

---

## Project Overview

Sentiment Analysis is a Natural Language Processing (NLP) technique used to determine the emotional tone of text.

In this project, a Simple Recurrent Neural Network (RNN) is used to classify IMDB movie reviews into two categories:

- **Positive Sentiment:** Reviews expressing a favorable opinion.
- **Negative Sentiment:** Reviews expressing an unfavorable opinion.

The project demonstrates how deep learning models can process sequential text data and generate sentiment predictions.

It includes model development notebooks, a saved trained model, and a Streamlit-based prediction interface.

## Problem Statement

Movie review platforms contain large amounts of user-generated text.

Manually analyzing every review can be time-consuming.

The objective of this project is to develop a deep learning solution that can automatically classify movie reviews based on their sentiment.

### Objectives

- Understand text preprocessing for NLP.
- Explore word embeddings.
- Implement a Simple RNN using TensorFlow/Keras.
- Work with the IMDB movie review dataset.
- Explore model training and hyperparameter tuning.
- Save and load a trained neural network.
- Build an interactive sentiment prediction application using Streamlit.

## Key Features

- Binary sentiment classification
- Simple RNN-based deep learning model
- Word embedding-based text representation
- IMDB dataset integration
- Text preprocessing and sequence padding
- Saved Keras model for inference
- Streamlit web application
- Interactive movie review input
- Positive/negative sentiment prediction
- Numerical prediction score display
- Separate notebooks for experimentation and prediction

## Technology Stack

| Category | Technology |
|----------|------------|
| Programming Language | Python |
| Deep Learning | TensorFlow, Keras |
| Natural Language Processing | Word Embeddings, Simple RNN |
| Dataset | IMDB Movie Reviews |
| Numerical Computing | NumPy |
| Model Storage | Keras H5 Format |
| Web Application | Streamlit |
| Development | Jupyter Notebook, VS Code |
| Version Control | Git, GitHub |

## Dataset

The project uses the **IMDB Movie Review Dataset**, available through TensorFlow/Keras.

The dataset is commonly used for binary sentiment classification.

### Dataset Characteristics

- Contains movie reviews.
- Reviews are associated with positive or negative sentiment labels.
- Text is represented using integer-encoded word sequences.
- The dataset provides a word index for converting words into numerical representations.

### Dataset Loading

```python
from tensorflow.keras.datasets import imdb

word_index = imdb.get_word_index()
```

The word index maps vocabulary words to integer identifiers.

These identifiers are used to prepare input sequences for the neural network.

## System Architecture

```text
             IMDB Movie Reviews
                      |
                      v
              Text Preprocessing
                      |
                      v
              Word-to-Index Mapping
                      |
                      v
                Sequence Padding
                      |
                      v
                 Embedding Layer
                      |
                      v
                 Simple RNN
                      |
                      v
              Binary Classification
                      |
                      v
              Positive / Negative
                      |
                      v
              Streamlit Interface
```

The diagram represents the conceptual model and inference workflow. Exact training-layer configurations should be checked in the training notebooks.

## Project Workflow

### 1. Data Preparation

The IMDB dataset provides movie reviews encoded as integer sequences.

Text must be converted into a numerical representation before it can be processed by the neural network.

### 2. Word Embeddings

Word embeddings represent words as dense numerical vectors.

Unlike simple integer identifiers, embeddings allow the neural network to learn useful representations of vocabulary items.

The repository includes `embedding.ipynb` for exploring embedding concepts.

### 3. Simple RNN

A Simple Recurrent Neural Network processes sequential information.

Unlike a standard feedforward neural network, an RNN maintains a hidden state that carries information from previous sequence elements.

This makes RNNs suitable for learning patterns in text sequences.

### 4. Model Experimentation

The repository contains notebooks for model development and hyperparameter tuning.

These include:

- `simplernn.ipynb`
- `hyperparametertuning_simplernn_completed.ipynb`

### 5. Model Saving

The trained model is stored as:

```text
simple_rnn_imdb.h5
```

This allows the model to be loaded for inference without retraining it each time the application starts.

### 6. Sentiment Prediction

The Streamlit application accepts a movie review, preprocesses it, and passes the resulting sequence to the trained model.

The model returns a numerical prediction score.

The application classifies the review as positive when the score exceeds 0.5 and negative otherwise.

## Model Architecture

The project uses a Simple RNN-based sentiment classification approach.

### Conceptual Components

**1. Input Representation**

Movie reviews are represented as sequences of integer-encoded words.

**2. Embedding Layer**

Transforms word identifiers into dense vector representations.

**3. Simple RNN Layer**

Processes sequential information and learns patterns from the review.

**4. Classification Output**

Produces a numerical output used for binary sentiment classification.

### Prediction Threshold

The application uses the following classification rule:

```python
sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
```

A prediction score above 0.5 is classified as positive.

A score of 0.5 or below is classified as negative.

The exact number of neurons, embedding dimensions, and training hyperparameters should be verified from the model development notebooks.

## Project Structure

```text
End-To-End-Deep-Learning-Project-With-Simple-RNN/
|
|-- .gitignore
|-- .vscode/
|
|-- embedding.ipynb
|   # Word embedding experiments
|
|-- simplernn.ipynb
|   # Simple RNN development notebook
|
|-- hyperparametertuning_simplernn_completed.ipynb
|   # Hyperparameter tuning experiments
|
|-- prediction.ipynb
|   # Prediction experimentation
|
|-- main.py
|   # Streamlit sentiment analysis application
|
|-- simple_rnn_imdb.h5
|   # Saved trained Keras model
|
|-- LICENSE
|
+-- README.md
```

## Installation and Setup

### Prerequisites

Ensure the following are installed:

- Python
- Git
- pip
- A compatible TensorFlow environment

### Step 1: Clone the Repository

```bash
git clone https://github.com/shammi00/End-To-End-Deep-Learning-Project-With-Simple-RNN.git

cd End-To-End-Deep-Learning-Project-With-Simple-RNN
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

Activate the virtual environment.

**Windows:**

```bash
venv\Scripts\activate
```

**Linux/macOS:**

```bash
source venv/bin/activate
```

### Step 3: Install Dependencies

The Streamlit application uses TensorFlow, NumPy, and Streamlit.

Install the required packages:

```bash
pip install tensorflow numpy streamlit
```

**Note:** A `requirements.txt` file is not currently included in the repository. Package versions should be selected based on compatibility with the saved model and your Python environment.

### Step 4: Verify Model File

Ensure the trained model is available in the project directory:

```text
simple_rnn_imdb.h5
```

The Streamlit application loads this file using:

```python
from tensorflow.keras.models import load_model

model = load_model("simple_rnn_imdb.h5")
```

## Running the Application

Start the Streamlit application using:

```bash
streamlit run main.py
```

Streamlit normally opens the application in your browser.

The default local address is:

```text
http://localhost:8501
```

### Using the Application

1. Open the Streamlit application.
2. Enter a movie review in the text area.
3. Click the **Classify** button.
4. The application preprocesses the review.
5. The trained RNN generates a prediction.
6. The sentiment and prediction score are displayed.

### Example Input

```text
This movie was amazing. The acting was excellent
and the story was very interesting.
```

The application will classify the review as positive or negative based on the trained model's output.

The example is illustrative; no specific prediction result is guaranteed.

## Prediction Process

### Step 1: Accept User Input

The application uses a Streamlit text area:

```python
user_input = st.text_area("Movie Review")
```

### Step 2: Preprocess Text

The existing application converts the review to lowercase, splits it into words, maps words to IMDB vocabulary indices, and pads the resulting sequence.

```python
def preprocess_text(text):
    words = text.lower().split()

    encoded_review = [
        word_index.get(word, 2) + 3
        for word in words
    ]

    padded_review = sequence.pad_sequences(
        [encoded_review],
        maxlen=500
    )

    return padded_review
```

The maximum sequence length used during inference is 500.

**Important:** This is the preprocessing implemented in `main.py`. Its consistency with the training-time preprocessing should be verified, particularly for punctuation and unknown words.

### Step 3: Generate Prediction

```python
prediction = model.predict(preprocessed_input)
```

### Step 4: Classify Sentiment

```python
sentiment = (
    "Positive"
    if prediction[0][0] > 0.5
    else "Negative"
)
```

### Step 5: Display Results

```python
st.write(f"Sentiment: {sentiment}")

st.write(
    f"Prediction Score: {prediction[0][0]}"
)
```

The numerical prediction score is the model output used for classification. It should not automatically be interpreted as a calibrated confidence value.

## Model Evaluation

Model evaluation is important for understanding how well a sentiment classification model generalizes to unseen reviews.

Common evaluation metrics include:

| Metric | Purpose |
|--------|---------|
| Accuracy | Overall proportion of correct predictions |
| Precision | Correctness of predicted positive reviews |
| Recall | Ability to identify actual positive reviews |
| F1-Score | Harmonic mean of precision and recall |
| Confusion Matrix | Distribution of correct and incorrect classifications |

### Recorded Performance

Verified evaluation results have not been included in this README.

Actual accuracy, precision, recall, and F1-score should be obtained from the model evaluation notebooks or by evaluating the saved model against an appropriate held-out test dataset.

No performance values are claimed without verification.

## Learning Outcomes

This project provided practical experience in:

- Understanding NLP and sentiment analysis
- Working with the IMDB movie review dataset
- Representing text using integer sequences
- Understanding word embeddings
- Implementing Simple Recurrent Neural Networks
- Exploring neural network hyperparameters
- Saving and loading Keras models
- Preprocessing text for inference
- Building interactive applications with Streamlit
- Integrating deep learning models with user interfaces
- Organizing a deep learning project using notebooks and application files

## Future Improvements

Potential improvements include:

- Comparing Simple RNN with LSTM and GRU models
- Evaluating model performance on held-out reviews
- Improving text tokenization and preprocessing consistency
- Adding a confusion matrix and classification report
- Providing clearer prediction score explanations
- Adding input validation and error handling
- Adding automated tests
- Creating a version-pinned `requirements.txt`
- Deploying the Streamlit application to a hosting platform
- Adding screenshots and a live application demo

## License

This repository includes a GNU General Public License Version 3 (GPL-3.0) license file.

See [LICENSE](LICENSE) for the full license text.

## Author

**Shammi Kumar Singh**

B.Tech — Artificial Intelligence & Machine Learning

Malla Reddy University

Expected Graduation: 2028

**GitHub:** https://github.com/shammi00

**LinkedIn:** https://www.linkedin.com/in/shammikumarsingh

## Repository Link

https://github.com/shammi00/End-To-End-Deep-Learning-Project-With-Simple-RNN

---

If you find this project useful, consider giving the repository a star!

**Built as part of my learning journey in Deep Learning, Natural Language Processing, and AI Application Development.**
