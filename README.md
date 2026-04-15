# Comparative Analysis of Word Representation Techniques for Multiclass Text Classification

This repository presents a comprehensive comparative study of different word representation techniques combined with classical machine learning and deep learning models for multiclass text classification. The project analyzes how various text vectorization methods impact classification performance, efficiency, and robustness across models.

The work includes exploratory data analysis (EDA), feature engineering, hyperparameter tuning, model training, and evaluation, supported by experimental results documented in the accompanying research paper.

---

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Word Representation Techniques](#word-representation-techniques)
- [Models Used](#models-used)
- [Results Summary](#results-summary)
- [How to Run](#how-to-run)
- [Tools and Technologies](#tools-and-technologies)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

---

## Overview

Natural Language Processing (NLP) relies heavily on how text is represented numerically. This project investigates and compares multiple word representation approaches, including traditional and neural embeddings, and evaluates them using classical machine learning classifiers and deep learning architectures for multiclass text classification.

The objective is to provide insight into how different representations affect model accuracy, generalization, and computational efficiency. The best performing combination was **Skip-gram embeddings with a Deep Neural Network (DNN)**, achieving 67.85% accuracy and a macro F1-score of 67.24%.

---

## Key Features

- Systematic comparison of Bag-of-Words (BoW), TF-IDF, GloVe, and Skip-gram embeddings.
- Evaluation across classical models (Logistic Regression, Naive Bayes, Random Forest) and deep learning architectures (DNN, RNN, GRU, LSTM, bidirectional variants).
- Full exploratory data analysis (EDA) including class distribution, text length, and word count statistics.
- Hyperparameter tuning for both classical and deep learning models.
- Reproducible Jupyter notebooks for preprocessing, training, and evaluation.
- Detailed quantitative results reported with accuracy, macro F1, weighted F1, and confusion matrices.

---

## Project Structure

```
.
├── README.md                                 # This file
├── Comparative_Analysis_of_Word_Representation_Techniques_with_Classical_ML_and_Deep_Learning_Models_for_Multiclass_Text_Classification.pdf  # Full paper
├── EDA_part_440_project.ipynb               # Exploratory data analysis and preprocessing
├── Hyper-parameter Tuning_CSE440_Group_Project.ipynb  # Hyperparameter optimization
├── Final Model Train and Test.ipynb          # Model training, testing, and evaluation pipelines
```

---

## Methodology

The project follows a structured pipeline:

1. **Data Loading and Cleaning** – The dataset consists of 279,999 training samples and 59,999 testing samples, balanced across ten categories (e.g., Society & Culture, Health, Computers & Internet).
2. **Exploratory Data Analysis (EDA)** – Class distribution, text length, and word count distributions were analyzed. No missing values were detected.
3. **Preprocessing** – Lowercasing, punctuation removal, tokenization, stopword removal, lemmatization, label encoding, and sequence padding (max length = 100 tokens).
4. **Feature Extraction** – Application of BoW, TF-IDF, GloVe (100‑dim), and custom Skip-gram embeddings (300‑dim).
5. **Model Training** – Classical and deep learning models trained with appropriate hyperparameters.
6. **Hyperparameter Tuning** – Systematic optimization using validation sets.
7. **Evaluation** – Accuracy, macro F1, weighted F1, and confusion matrices.

---

## Word Representation Techniques

| Technique | Description |
|-----------|-------------|
| **Bag-of-Words (BoW)** | Sparse vector of token counts; ignores word order but simple and interpretable. |
| **TF-IDF** | Term Frequency–Inverse Document Frequency; reduces influence of common words. |
| **GloVe** | Pre-trained 100‑dimensional embeddings capturing global word co‑occurrence statistics. |
| **Skip-gram (Word2Vec)** | Custom 300‑dimensional embeddings trained on the dataset to capture domain‑specific semantics. |

---

## Models Used

### Classical Machine Learning Models
- Logistic Regression
- Naive Bayes (Multinomial)
- Random Forest

### Deep Learning Models
- Deep Neural Network (DNN) – fully connected feed‑forward network with ReLU and dropout.
- SimpleRNN
- Gated Recurrent Unit (GRU)
- Long Short-Term Memory (LSTM)
- Bidirectional variants of SimpleRNN, GRU, and LSTM

All deep learning models were implemented in Keras with TensorFlow backend, using the Adam optimizer and categorical cross‑entropy loss.

---

## Results Summary

The table below presents the best performance for each representation category:

| Representation | Model | Accuracy | F1-Macro | F1-Weighted |
|----------------|-------|----------|-----------|---------------|
| BoW            | DNN   | 0.623    | 0.621     | 0.621         |
| TF-IDF         | Logistic Regression | 0.637 | 0.634 | 0.634 |
| GloVe          | DNN   | 0.662    | 0.652     | 0.652         |
| **Skip-gram**  | **DNN** | **0.679** | **0.672** | **0.672** |

Key observations:
- Skip-gram DNN achieved the highest overall performance (67.9% accuracy).
- TF-IDF with Logistic Regression performed best among classical combinations (63.7% accuracy).
- Pre‑trained GloVe embeddings improved over TF‑IDF but were inferior to task‑specific Skip‑gram embeddings.
- Recurrent models (GRU, LSTM) provided moderate gains but were computationally expensive and suffered from vanishing gradients in SimpleRNN.
- Random Forest performed poorly on sparse BoW/TF‑IDF features.

Detailed confusion matrices and per‑class metrics are available in the included PDF paper.

---

## How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/RummanShahriar/NLP-Comparative-Analysis-of-Word-Representation-Techniques.git
   cd NLP-Comparative-Analysis-of-Word-Representation-Techniques
   ```

2. **Install dependencies**
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn nltk tensorflow torch
   ```

3. **Run the Jupyter notebooks in order**
   - `EDA_part_440_project.ipynb` – data inspection and preprocessing.
   - `Hyper-parameter Tuning_CSE440_Group_Project.ipynb` – hyperparameter optimization.
   - `Final Model Train and Test.ipynb` – training, evaluation, and result generation.

> Note: The dataset is not included in this repository due to size constraints. Please place your own text classification dataset in the appropriate format (e.g., CSV with `text` and `label` columns) or update the notebook paths accordingly.

---

## Tools and Technologies

- **Language:** Python 3.x
- **Libraries:** NumPy, Pandas, scikit-learn, NLTK, TensorFlow, PyTorch (optional)
- **Visualization:** Matplotlib, Seaborn
- **Environment:** Jupyter Notebook

---

## Citation

If you use this code or the findings from this study in your research, please cite the accompanying paper:

```bibtex
@article{islam2025comparative,
  title={Comparative Analysis of Word Representation Techniques with Classical ML and Deep Learning Models for Multiclass Text Classification},
  author={Islam, Mehrabul and Shahriar, Md. Rumman},
  journal={BRAC University Technical Report},
  year={2025}
}
```

---

## Acknowledgments

The authors thank the Department of Computer Science and Engineering at BRAC University for providing the computational resources and academic support that made this research possible.

---

## Contact

**Md. Rumman Shahriar**  
GitHub: [RummanShahriar](https://github.com/RummanShahriar)  
Email: md.rumman.shahriar@g.bracu.ac.bd  

**Mehrabul Islam**  
Email: mehrabul.islam@g.bracu.ac.bd

For questions or issues, please open an issue in this repository.
