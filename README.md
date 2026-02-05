# NLP Comparative Analysis of Word Representation Techniques

This repository presents a comprehensive comparative study of different word representation techniques combined with classical machine learning and deep learning models for multiclass text classification. The project analyzes how various text vectorization methods impact classification performance, efficiency, and robustness across models.

The work includes exploratory data analysis (EDA), feature engineering, hyperparameter tuning, model training, and evaluation, supported by experimental results documented in the accompanying research paper.

---

## Overview

Natural Language Processing (NLP) relies heavily on how text is represented numerically. This project investigates and compares multiple word representation approaches, including traditional and neural embeddings, and evaluates them using classical machine learning classifiers and deep learning architectures for multiclass text classification.

The objective is to provide insight into how different representations affect model accuracy, generalization, and computational efficiency.

---

## Project Structure

The repository contains the following main components:

- EDA_part_440_project.ipynb  
  Performs exploratory data analysis, dataset inspection, visualization, and preprocessing.

- Hyper-parameter Tuning_CSE440_Group_Project.ipynb  
  Conducts systematic hyperparameter optimization for both classical and deep learning models.

- Final Model Train and Test.ipynb  
  Implements model training, testing, and evaluation pipelines.

- Comparative_Analysis_of_Word_Representation_Techniques_with_Classical_ML_and_Deep_Learning_Models_for_Multiclass_Text_Classification.pdf  
  The full project paper describing methodology, experiments, and results.

- README.md  
  Project documentation.

---

## Objectives

- Compare multiple word representation techniques for text classification.  
- Analyze performance across classical machine learning and deep learning models.  
- Evaluate accuracy, precision, recall, and F1-score for multiclass problems.  
- Study the effect of feature representations on model efficiency.  
- Provide reproducible experiments for further research.

---

## Word Representation Techniques

The project explores a range of text vectorization approaches, including:

- Bag of Words (BoW).  
- TF-IDF.  
- Word embedding based representations.  
- Token and sequence encodings for deep learning models.

These representations are evaluated across different model families to highlight strengths and weaknesses.

---

## Models Used

### Classical Machine Learning Models

- Logistic Regression.  
- Naive Bayes.  
- Support Vector Machine (SVM).  
- Random Forest.  
- K-Nearest Neighbors.

### Deep Learning Models

- Feedforward Neural Networks.  
- Recurrent Neural Networks (RNN, LSTM, GRU where applicable).  
- Other neural architectures implemented in the notebooks.

---

## Methodology

1. Data loading and cleaning.  
2. Exploratory Data Analysis (EDA).  
3. Text preprocessing and normalization.  
4. Feature extraction using multiple word representations.  
5. Model training and selection.  
6. Hyperparameter tuning.  
7. Evaluation using multiclass metrics.  
8. Comparative analysis of results.

---

## Evaluation Metrics

Models are evaluated using:

- Accuracy.  
- Precision.  
- Recall.  
- F1-score.  
- Confusion Matrix.

These metrics provide insight into both overall and per-class performance.

---

## How to Run

1. Clone the repository:

git clone https://github.com/RummanShahriar/NLP-Comparative-Analysis-of-Word-Representation-Techniques.git  
cd NLP-Comparative-Analysis-of-Word-Representation-Techniques  

2. Install required dependencies:

pip install numpy pandas scikit-learn matplotlib seaborn nltk tensorflow torch  

3. Run the notebooks in order:

- EDA_part_440_project.ipynb  
- Hyper-parameter Tuning_CSE440_Group_Project.ipynb  
- Final Model Train and Test.ipynb  

---

## Results

The experiments show how different word representations significantly influence classification performance. Traditional approaches such as TF-IDF perform competitively with classical models, while embedding-based approaches generalize better when paired with deep learning models.

Detailed quantitative results and discussion are available in the included PDF paper.

---

## Tools and Technologies

- Language: Python 3.x.  
- Libraries: NumPy, Pandas, scikit-learn, NLTK, TensorFlow, PyTorch.  
- Visualization: Matplotlib, Seaborn.  
- Environment: Jupyter Notebook.

---

## Applications

- Text classification systems.  
- News categorization.  
- Sentiment analysis.  
- Topic classification.  
- NLP research and benchmarking.

---
