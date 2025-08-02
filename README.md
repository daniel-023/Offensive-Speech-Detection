# Offensive Speech Detection

## Project Introduction
This project focuses on comparing the performance of three different vectorisation methods—TF-IDF, Word2Vec, and Sentence Transformers—on a binary classification task (offensive speech vs. non-offensive speech). The performance of these vectorisation methods is evaluated using three different classifiers: Logistic Regression, Support Vector Machine (SVM), and Random Forest. 

## Data
- 8326 manually labelled tweets  
- Dataset: [Hate Speech and Offensive Language Dataset](https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset) 

![Class Distribution Plot](plots/class_distribution_plot.png)

## Results

|                       |                     | Accuracy | Precision |  Recall  |    F1    |
|----------------------:|--------------------:|---------:|----------:|---------:|---------:|
|                TF-IDF | Logistic Regression | 0.939106 |  0.972814 | 0.903632 | 0.936859 |
|                       |                 SVM | 0.948955 |  0.975104 | 0.921582 | 0.947527 |
|                       |       Random Forest | 0.946912 |  0.966593 | 0.925818 | 0.945685 |
|              Word2Vec | Logistic Regression | 0.843141 |  0.848128 | 0.836022 | 0.841914 |
|                       |                 SVM | 0.855872 |  0.865597 | 0.842845 | 0.853873 |
|                       |       Random Forest | 0.873286 |  0.893094 | 0.848054 | 0.869846 |
| Sentence Transformers | Logistic Regression | 0.919169 |  0.937130 | 0.898897 | 0.917542 |
|                       |                 SVM | 0.922171 |  0.944822 | 0.897017 | 0.920205 |
|                       |       Random Forest | 0.897670 |  0.930286 | 0.860301 | 0.893745 |

### Heat Maps
| Accuracy                                            | Precision                                           | 
| --------------------------------------------------- | --------------------------------------------------- | 
| ![Accuracy Heat Map](plots/accuracy_heatmap.png)    | ![Precision Heat Map](plots/precision_heatmap.png)  | 

| Recall                                              | F1                                                  |
| --------------------------------------------------- | --------------------------------------------------- |
| ![Recall Heat Map](plots/recall_heatmap.png)        | ![F1 Heat Map](plots/f1_heatmap.png)                |

### Radar Charts
![Logistic Regression Radar](plots/logreg_radar.png) 
![SVM Radar](plots/SVM_radar.png) 
![RF Radar](plots/RF_radar.png) 

## 01_data_exploration_and_preprocessing.ipynb
### Objective: Explore and preprocess the dataset to prepare it for further analysis.
- Contents:
  - Data Loading: Import and inspect the dataset.
  - Class Distribution: Analyze the distribution of target labels.
  - Data Cleaning: Handle missing values, remove usernames, and preprocess text (e.g., lowercasing, tokenization).
## 02_feature_extraction.ipynb
### Objective: Generate feature representations using various vectorization techniques.
- Contents:
  - TF-IDF Vectorization: Compute TF-IDF features for the text data.
  - Word2Vec Embeddings: Train Word2Vec and generate sentence-level embeddings.
  - Sentence Transformers: Extract embeddings using pre-trained Sentence Transformer models.
  - Feature Storage: Save generated feature matrices for model training and evaluation.
## 03_model_training_and_evaluation.ipynb
### Objective: Train and evaluate machine learning models using different feature representations.
- Contents:
  - Model Training: Train Logistic Regression, SVM, and Random Forest classifiers with TF-IDF, Word2Vec, and Sentence Transformer features.
  - Cross-Validation: Apply k-fold cross-validation to assess model performance.
  - Evaluation Metrics: Compute accuracy, precision, recall, and F1-score.
  - Results Storage: Save structured performance results for further analysis.
## 04_results_visualisation_and_interpretation.ipynb
### Objective: Visualize and interpret model performance across different vectorization techniques.
- Contents:
  - Radar Charts: Compare classifier performance across different feature representations.
  - Heatmaps: Visualize performance metrics (accuracy, precision, recall, F1-score).
  - Analysis: Interpret results, discussing strengths and weaknesses of each approach.
