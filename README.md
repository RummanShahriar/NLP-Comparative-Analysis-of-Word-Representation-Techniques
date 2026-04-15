# Comparative Analysis of Word Representation Techniques with Classical Machine Learning and Deep Learning Models for Multiclass Text Classification

This repository presents a comprehensive, systematic comparative study of different word representation techniques combined with classical machine learning and deep learning models for multiclass text classification. The project investigates how various text vectorization methods, ranging from sparse statistical representations to dense neural embeddings, impact classification performance, computational efficiency, and model robustness across a diverse set of classifiers.

The work includes exploratory data analysis (EDA), text preprocessing, feature engineering, hyperparameter tuning, model training, and thorough evaluation. All experimental results and detailed discussions are documented in the accompanying research paper included in this repository.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Dataset Description](#dataset-description)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Preprocessing Pipeline](#preprocessing-pipeline)
- [Word Representation Techniques](#word-representation-techniques)
  - [Bag-of-Words (BoW)](#bag-of-words-bow)
  - [TF-IDF](#tf-idf)
  - [GloVe Embeddings](#glove-embeddings)
  - [Skip-gram (Word2Vec) Embeddings](#skip-gram-word2vec-embeddings)
- [Model Architectures](#model-architectures)
  - [Classical Machine Learning Models](#classical-machine-learning-models)
  - [Deep Neural Networks (DNN)](#deep-neural-networks-dnn)
  - [Recurrent Neural Networks (RNN, GRU, LSTM)](#recurrent-neural-networks-rnn-gru-lstm)
  - [Bidirectional Variants](#bidirectional-variants)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Evaluation Metrics](#evaluation-metrics)
- [Results and Discussion](#results-and-discussion)
  - [Performance Comparison Tables](#performance-comparison-tables)
  - [Key Findings](#key-findings)
  - [Visualization](#visualization)
- [How to Run the Code](#how-to-run-the-code)
  - [Installation](#installation)
  - [Execution Order](#execution-order)
- [Tools and Technologies](#tools-and-technologies)
- [Applications](#applications)
- [Limitations and Future Work](#limitations-and-future-work)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

---

## Overview

Text classification is a fundamental task in Natural Language Processing (NLP) with widespread applications including sentiment analysis, spam detection, topic labeling, and information retrieval. The effectiveness of any text classification system depends not only on the choice of the classifier but critically on how the raw text is transformed into numerical features that machine learning algorithms can process.

This project provides a rigorous comparative analysis of four distinct word representation techniques:

1. **Bag-of-Words (BoW)** , a sparse, count-based representation.
2. **TF-IDF** , a normalized frequency representation that down-weights common terms.
3. **GloVe** , pre-trained static dense embeddings based on global word co-occurrence statistics.
4. **Skip-gram (Word2Vec)** , task-specific dense embeddings trained from scratch on the given dataset.

These representations are evaluated across a wide spectrum of models, from simple linear classifiers (Logistic Regression, Naive Bayes) to ensemble methods (Random Forest) and advanced deep learning architectures (DNN, RNN, GRU, LSTM, and their bidirectional variants). The goal is to identify which combinations yield optimal performance in terms of accuracy, F1-score, and computational efficiency for multiclass text classification.

The best performing configuration, **Skip-gram embeddings with a Deep Neural Network**, achieved an accuracy of 67.85% and a macro F1-score of 67.24%, outperforming all other combinations.

---

## Key Features

- **Comprehensive representation comparison:** BoW, TF-IDF, GloVe, and Skip-gram embeddings are systematically compared under identical experimental conditions.
- **Diverse model suite:** Includes classical machine learning (Logistic Regression, Naive Bayes, Random Forest) and deep learning (DNN, SimpleRNN, GRU, LSTM, bidirectional variants).
- **Thorough exploratory data analysis:** Class distribution, text length statistics, word count distributions, and missing value detection.
- **Robust preprocessing pipeline:** Lowercasing, punctuation removal, stopword removal, lemmatization, tokenization, and sequence padding.
- **Hyperparameter tuning:** Systematic optimization of model parameters using validation sets.
- **Extensive evaluation metrics:** Accuracy, macro F1, weighted F1, confusion matrices, and per-class classification reports.
- **Reproducible Jupyter notebooks:** Full end-to-end pipeline from EDA to final evaluation.
- **Accompanying research paper:** Detailed methodology, results, and discussion in PDF format.

---

## Project Structure

```
.
├── README.md                                 # This documentation file
├── Comparative_Analysis_of_Word_Representation_Techniques_with_Classical_ML_and_Deep_Learning_Models_for_Multiclass_Text_Classification.pdf  # Full project paper
├── EDA_part_440_project.ipynb               # Exploratory data analysis, preprocessing, and visualization
├── Hyper-parameter Tuning_CSE440_Group_Project.ipynb  # Systematic hyperparameter optimization for all models
├── Final Model Train and Test.ipynb          # Model training, testing, evaluation, and result aggregation
```

Each notebook is self-contained and well-commented. It is recommended to run them in the order listed above.

---

## Dataset Description

The dataset used in this project consists of a large collection of labeled text documents balanced across ten distinct categories. The categories include:

- Society & Culture
- Health
- Computers & Internet
- (Additional categories as present in the original dataset)

**Dataset statistics:**

| Split          | Number of Samples |
|----------------|-------------------|
| Training set   | 279,999           |
| Testing set    | 59,999            |

**Characteristics:**

- No missing values in either training or testing sets.
- Class distribution is nearly uniform (approximately 10% per class), making it suitable for multi-class classification without severe class imbalance issues.
- Documents vary in length; the majority contain between 200 and 700 characters and 30 to 120 words.
- A long tail of longer documents (over 1,000 words) exists, providing variety and challenge for sequence-based models.

The dataset is not included in this repository due to size constraints. To reproduce the experiments, place your own CSV file with `text` and `label` columns, or update the data loading paths in the notebooks accordingly.

---

## Exploratory Data Analysis (EDA)

The `EDA_part_440_project.ipynb` notebook performs a thorough initial analysis of the dataset, which is critical for informing preprocessing decisions and model selection. The following analyses are included:

1. **Class Distribution Analysis:** Bar charts showing the frequency of each class in both training and testing sets. The near-uniform distribution confirms that no class weighting or resampling is required.

2. **Text Length Distribution:** Histograms of document lengths (number of characters) reveal that most documents are short to medium in length, with a long tail of longer documents. This informed the choice of sequence length (100 tokens) for deep learning models.

3. **Word Count Distribution:** Similar to text length, the distribution of word counts per document shows a right-skewed pattern, consistent with Zipf's law. This analysis helps set appropriate vocabulary size limits.

4. **Missing Value Check:** Confirms that both training and testing sets have no missing values, eliminating the need for imputation.

5. **Sample Inspection:** Random samples of text are displayed to understand the nature of the content, typical vocabulary, and potential noise (e.g., special characters, HTML tags, etc.).

Visualizations from EDA are saved within the notebook and referenced in the PDF paper.

---

## Preprocessing Pipeline

Based on insights from EDA, a consistent preprocessing pipeline was applied to both training and testing sets. The pipeline ensures that raw text is converted into a clean, standardized format suitable for feature extraction and model training. Steps are as follows:

1. **Lowercasing:** All text is converted to lowercase to ensure that words like "Apple" and "apple" are treated as the same token, reducing vocabulary size and improving generalization.

2. **Punctuation and Special Character Removal:** Non-alphanumeric characters (e.g., commas, periods, exclamation marks, brackets, HTML tags) are stripped from the text. This reduces noise and prevents the model from learning spurious patterns.

3. **Whitespace Normalization:** Extra spaces, tabs, and newlines are collapsed into single spaces, ensuring consistent tokenization.

4. **Tokenization:** Sentences are split into individual words (tokens) using NLTK's word tokenizer. This produces a list of tokens for each document.

5. **Stopword Removal:** Common English stopwords (e.g., "the", "and", "is", "of", "to") are removed using NLTK's stopword corpus. Stopwords carry little semantic information for classification and their removal reduces dimensionality.

6. **Lemmatization:** Each token is reduced to its base or dictionary form (lemma) using the WordNet lemmatizer. For example, "running" becomes "run", "better" becomes "good". Unlike stemming, lemmatization preserves valid word forms, which is beneficial for semantic understanding.

7. **Label Encoding:** Class labels (originally strings such as "Health", "Technology") are converted to integer indices (0, 1, 2, ...) for compatibility with machine learning algorithms.

8. **Sequence Preparation (for deep learning models):** The Keras `Tokenizer` is used to convert text into sequences of integer indices, with a vocabulary size limited to 10,000 most frequent words to manage memory. Sequences are padded or truncated to a fixed length of 100 tokens. This ensures uniform input size for recurrent networks.

9. **Subsampling (optional):** To reduce computational load during initial experiments, 50% of the training data can be used. This preserves the class distribution and provides a representative subset for hyperparameter tuning.

All preprocessing steps are implemented in the Jupyter notebooks and can be modified or extended as needed.

---

## Word Representation Techniques

After preprocessing, the clean text is transformed into numerical feature vectors using four distinct techniques. Each technique has its own strengths and weaknesses, and the project evaluates them side by side.

### Bag-of-Words (BoW)

- **Description:** Each document is represented as a sparse vector where each dimension corresponds to a unique word (token) in the entire corpus, and the value is the frequency of that word in the document. Word order is ignored, hence the name "bag of words".
- **Implementation:** Scikit-learn's `CountVectorizer` is used, with parameters: `max_features=10000` (limit vocabulary size), `ngram_range=(1,2)` to include both unigrams and bigrams.
- **Advantages:** Simple, fast, interpretable, and works well for many baseline tasks.
- **Disadvantages:** High-dimensional sparse vectors, no semantic information, cannot capture word order or context.

### TF-IDF (Term Frequency-Inverse Document Frequency)

- **Description:** An extension of BoW that scales word frequencies by the inverse of the number of documents containing that word. This reduces the weight of common words (e.g., "the", "a") and increases the weight of rare but informative words.
- **Implementation:** Scikit-learn's `TfidfVectorizer` with `max_features=10000`, `ngram_range=(1,2)`, and `sublinear_tf=True` to use log scaling.
- **Advantages:** Better than BoW for many text classification tasks because it down-weights common terms. Still fast and interpretable.
- **Disadvantages:** Still sparse, still ignores word order and semantics.

### GloVe Embeddings

- **Description:** Global Vectors for Word Representation (GloVe) is an unsupervised learning method that pre-trains dense word vectors on large corpora (e.g., Wikipedia, Gigaword). The vectors capture semantic relationships: words with similar meanings have similar vectors.
- **Implementation:** Pre-trained 100-dimensional GloVe vectors are loaded. A Keras `Embedding` layer is initialized with these weights and frozen (non-trainable) during training. Words not found in the GloVe vocabulary are assigned random vectors.
- **Advantages:** Dense representations (100 dimensions), capture semantic and syntactic similarities, transfer learning from large corpora.
- **Disadvantages:** Static (same vector for a word regardless of context), may not capture domain-specific nuances if the pre-training corpus differs from the target dataset.

### Skip-gram (Word2Vec) Embeddings

- **Description:** The Skip-gram model (a variant of Word2Vec) learns word embeddings by training a shallow neural network to predict surrounding context words given a target word. Unlike GloVe, these embeddings are trained **from scratch** on the specific dataset used in this project.
- **Implementation:** Custom training using the Gensim library. Parameters: `vector_size=300`, `window=5`, `min_count=5`, `sg=1` (skip-gram), `epochs=20`. The resulting 300-dimensional embeddings are then used to initialize an `Embedding` layer in Keras.
- **Advantages:** Task-specific, captures domain semantics more effectively than generic pre-trained embeddings. Dense and relatively low-dimensional.
- **Disadvantages:** Requires training from scratch (computationally more expensive than loading pre-trained vectors), performance depends on dataset size and quality.

---

## Model Architectures

### Classical Machine Learning Models

These models are trained on the sparse BoW and TF-IDF representations. They serve as baselines and provide insights into how well traditional methods perform on this task.

- **Logistic Regression:** A linear model that uses the logistic (sigmoid) function to estimate class probabilities. Despite its simplicity, it often performs surprisingly well on high-dimensional sparse text features. Optimized using the `liblinear` solver with L2 regularization.

- **Naive Bayes (Multinomial):** A probabilistic classifier based on Bayes' theorem with the assumption of conditional independence between features. The multinomial variant is particularly suited for discrete counts (word frequencies). It is fast and works well with BoW features.

- **Random Forest:** An ensemble method that builds multiple decision trees and aggregates their predictions via majority voting. While capable of capturing non-linear interactions, it tends to perform poorly on high-dimensional sparse features (curse of dimensionality) and is slower than linear models.

### Deep Neural Networks (DNN)

A fully connected feed-forward neural network was implemented for the dense embedding representations (GloVe and Skip-gram). The architecture is as follows:

- **Input layer:** Accepts the document embedding (either mean of word embeddings or concatenated features depending on representation).
- **Hidden layers:** Two dense layers with 128 and 64 units respectively, each followed by ReLU activation and batch normalization.
- **Dropout:** A dropout rate of 0.5 after each hidden layer to prevent overfitting.
- **Output layer:** Dense layer with 10 units (number of classes) and softmax activation.
- **Loss function:** Categorical cross-entropy.
- **Optimizer:** Adam with learning rate 0.001.
- **Training:** Early stopping (patience = 5) and model checkpointing.

### Recurrent Neural Networks (RNN, GRU, LSTM)

Recurrent models are designed to process sequential data (i.e., the ordered sequence of word embeddings in a document). They can capture contextual dependencies that feed-forward networks miss.

- **SimpleRNN:** The basic recurrent unit that maintains a hidden state across time steps. However, it suffers from the vanishing gradient problem, making it difficult to learn long-range dependencies. This was reflected in poor performance (accuracy ~0.25).

- **Gated Recurrent Unit (GRU):** An improved recurrent architecture that uses update and reset gates to control information flow. It is less prone to vanishing gradients and computationally more efficient than LSTM.

- **Long Short-Term Memory (LSTM):** A more complex recurrent architecture with input, forget, and output gates. LSTMs are widely used for sequence modeling and can capture long-term dependencies effectively.

- **Bidirectional Variants:** For both GRU and LSTM, bidirectional versions were implemented. A bidirectional RNN processes the sequence in both forward and reverse directions and concatenates the hidden states, allowing the model to consider past and future context simultaneously.

All recurrent models use an embedding layer (initialized with GloVe or Skip-gram weights, or trainable from scratch), followed by the recurrent layer(s), a dropout layer, and a dense softmax output layer.

---

## Hyperparameter Tuning

The notebook `Hyper-parameter Tuning_CSE440_Group_Project.ipynb` performs systematic hyperparameter optimization using grid search and random search strategies. Key hyperparameters tuned include:

- For classical models:
  - Logistic Regression: `C` (inverse regularization strength), solver type.
  - Random Forest: `n_estimators`, `max_depth`, `min_samples_split`.
  - Naive Bayes: smoothing parameter `alpha`.

- For deep learning models:
  - Number of layers and units per layer.
  - Dropout rates.
  - Learning rate and optimizer settings.
  - Batch size and number of epochs.
  - Sequence length and vocabulary size.

The tuning process uses a validation split of the training data (80% train, 20% validation) and monitors accuracy. The best hyperparameters are then used for final evaluation on the test set.

---

## Evaluation Metrics

All models are evaluated using the following metrics to ensure a comprehensive assessment:

- **Accuracy:** The proportion of correctly classified samples over the total number of samples. Useful for balanced datasets.

- **Macro F1-score:** The unweighted average of F1-scores across all classes. Treats all classes equally, regardless of their frequency. This is important for multi-class problems.

- **Weighted F1-score:** The average of F1-scores weighted by the number of true instances per class. Accounts for class imbalance if present (though this dataset is balanced).

- **Confusion Matrix:** A table showing true vs. predicted class labels, which helps identify which classes are frequently confused with each other.

- **Classification Report:** Per-class precision, recall, and F1-score.

These metrics are computed using Scikit-learn's `classification_report` and `confusion_matrix` functions.

---

## Results and Discussion

### Performance Comparison Tables

The following table summarizes the best performance achieved by each combination of representation and model. Only the top-performing models are shown for clarity.

| Representation | Model | Accuracy | F1-Macro | F1-Weighted |
|----------------|-------|----------|-----------|---------------|
| BoW | DNN | 0.623 | 0.621 | 0.621 |
| BoW | Logistic Regression | 0.612 | 0.609 | 0.609 |
| TF-IDF | Logistic Regression | **0.637** | **0.634** | **0.634** |
| TF-IDF | DNN | 0.635 | 0.631 | 0.631 |
| GloVe | DNN | 0.662 | 0.652 | 0.652 |
| GloVe | GRU | 0.642 | 0.637 | 0.637 |
| GloVe | LSTM | 0.630 | 0.627 | 0.627 |
| Skip-gram | **DNN** | **0.679** | **0.672** | **0.672** |
| Skip-gram | GRU | 0.616 | 0.611 | 0.611 |
| Skip-gram | LSTM | 0.612 | 0.607 | 0.607 |

Recurrent models with GloVe and Skip-gram performed worse than the DNN, likely due to the relatively short sequence length (100 tokens) and the fact that the dataset does not require long-range dependencies. SimpleRNN performed very poorly (accuracy ~0.25) due to vanishing gradients.

### Key Findings

1. **Skip-gram + DNN is the best combination** with 67.9% accuracy and 67.2% macro F1. This demonstrates that task-specific embeddings trained from scratch capture domain semantics more effectively than generic pre-trained embeddings.

2. **TF-IDF with Logistic Regression** is the best classical combination (63.7% accuracy), outperforming BoW and Random Forest. This confirms the value of TF-IDF weighting and the suitability of linear models for sparse text features.

3. **GloVe embeddings improve over TF-IDF** when paired with a DNN (66.2% vs 63.7%), showing that dense semantic representations help classification. However, GloVe underperforms compared to Skip-gram, indicating that domain-specific training is beneficial.

4. **Recurrent models did not outperform the DNN** in this task. This may be due to the relatively short document lengths (most under 120 words) and the fact that word order may not be as critical for topic classification as for tasks like sentiment analysis. Additionally, the computational cost of training RNNs is much higher.

5. **SimpleRNN is unsuitable** for this task due to vanishing gradients, as evidenced by near-random performance.

6. **Bidirectional variants** did not provide significant gains over unidirectional GRU/LSTM, likely because future context is not as informative as past context for topic classification.

### Visualization

The notebooks generate the following visualizations:

- **Accuracy comparison bar chart** (Figure 7 in the paper) showing all model-representation combinations side by side.
- **Training curves** (Figure 8) for the best model (Skip-gram DNN), illustrating loss and accuracy over epochs.
- **Confusion matrix** (Figure 9) for the Skip-gram DNN, showing high diagonal values and very few off-diagonal misclassifications.

These plots are saved as PNG files within the notebook execution environment.

---

## How to Run the Code

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/RummanShahriar/NLP-Comparative-Analysis-of-Word-Representation-Techniques.git
   cd NLP-Comparative-Analysis-of-Word-Representation-Techniques
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Linux/Mac
   venv\Scripts\activate      # On Windows
   ```

3. **Install required packages:**
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn nltk tensorflow torch gensim
   ```

   Note: `torch` is optional; the primary deep learning framework is TensorFlow/Keras. Gensim is required for training Skip-gram embeddings.

4. **Download NLTK data (if not already present):**
   Run the following in a Python shell or in the first notebook cell:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   nltk.download('omw-1.4')
   ```

### Execution Order

Run the Jupyter notebooks in the following sequence:

1. **`EDA_part_440_project.ipynb`**  
   - Loads the dataset.  
   - Performs EDA (class distributions, length analysis, etc.).  
   - Applies preprocessing (cleaning, tokenization, stopwords, lemmatization).  
   - Saves cleaned data for subsequent notebooks.

2. **`Hyper-parameter Tuning_CSE440_Group_Project.ipynb`**  
   - Loads preprocessed data.  
   - Defines hyperparameter search spaces for each model.  
   - Runs grid search / random search.  
   - Saves best hyperparameters to disk.

3. **`Final Model Train and Test.ipynb`**  
   - Loads best hyperparameters.  
   - Trains each model (BoW, TF-IDF, GloVe, Skip-gram) with corresponding classical and deep learning models.  
   - Evaluates on the test set.  
   - Generates performance tables, confusion matrices, and plots.  
   - Saves results and trained models (optional).

> **Note:** The dataset is not provided. You must place your own CSV file (with columns `text` and `label`) in the appropriate directory and update the file path in the notebooks. Alternatively, you can adapt the notebooks to load data from any source.

---

## Tools and Technologies

- **Programming Language:** Python 3.8+
- **Data Processing:** NumPy, Pandas
- **Text Preprocessing:** NLTK (tokenization, stopwords, lemmatization)
- **Classical ML & TF-IDF/BoW:** Scikit-learn
- **Deep Learning:** TensorFlow 2.x with Keras API
- **Word Embeddings (Skip-gram):** Gensim
- **Visualization:** Matplotlib, Seaborn
- **Environment:** Jupyter Notebook, Google Colab (optional)

---

## Applications

The findings of this project have practical implications for various NLP applications:

- **News Categorization:** Automatically assigning topics to news articles.
- **Sentiment Analysis:** Though not directly evaluated, the same methodology applies.
- **Spam Detection:** Binary classification but can be extended.
- **Customer Support Ticket Routing:** Classifying incoming messages into departments.
- **Academic Paper Classification:** Grouping research papers by field.
- **Information Retrieval and Search:** Improving document indexing.

The comparative insights help practitioners choose the right representation-model pair based on their accuracy and efficiency requirements.

---

## Limitations and Future Work

**Limitations of the current study:**

1. **Computational constraints:** Training recurrent neural networks on the full dataset (280k samples) was resource-intensive. Subsampling was used during hyperparameter tuning, which might have affected the optimal hyperparameters.

2. **Sequence length limitation:** A fixed length of 100 tokens was used. Longer documents were truncated, potentially losing information.

3. **No transformer models:** State-of-the-art transformer architectures (e.g., BERT, RoBERTa, DistilBERT) were not included due to computational constraints. These would likely achieve higher accuracy but at greater computational cost.

4. **SimpleRNN failure:** The poor performance of SimpleRNN is well-known, but it was included as a baseline to demonstrate the vanishing gradient problem empirically.

**Future work directions:**

1. **Integrate transformer models:** Fine-tune a lightweight transformer (e.g., DistilBERT) on the same dataset and compare performance.

2. **Data augmentation:** Use back-translation or synonym replacement to increase dataset size and improve robustness.

3. **Ensemble methods:** Combine predictions from multiple models (e.g., Skip-gram DNN + TF-IDF Logistic Regression) to boost accuracy.

4. **Long document handling:** Experiment with hierarchical attention networks or document-level transformers.

5. **Cross-dataset validation:** Test the best models on other text classification benchmarks to measure generalization.

6. **Deployment optimization:** Convert the best model to TensorFlow Lite or ONNX for edge deployment.

---

## Citation

If you use this code, methodology, or findings in your research, please cite the accompanying paper:

```bibtex
@article{islam2025comparative,
  title={Comparative Analysis of Word Representation Techniques with Classical ML and Deep Learning Models for Multiclass Text Classification},
  author={Islam, Mehrabul and Shahriar, Md. Rumman},
  journal={BRAC University Technical Report},
  year={2025}
}
```

For the dataset (if you are using the same public dataset), please cite the original source as indicated in the paper.

---

## Acknowledgments

The authors gratefully acknowledge the Department of Computer Science and Engineering at BRAC University for providing the computational resources, academic guidance, and supportive research environment that made this project possible. Special thanks to the faculty members and teaching assistants for their valuable feedback during the course.

---

## Contact

**Md. Rumman Shahriar**  
GitHub: [RummanShahriar](https://github.com/RummanShahriar)  
Email: md.rumman.shahriar@g.bracu.ac.bd  

**Mehrabul Islam**  
Email: mehrabul.islam@g.bracu.ac.bd  

For questions, issues, or collaboration inquiries, please open an issue in this GitHub repository or contact the authors directly via email.
