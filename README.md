# 🧠 Financial Sentiment Classification with Traditional ML and LLMs

This project focuses on sentiment classification in financial texts using a mix of traditional machine learning (Logistic Regression, XGBoost) and modern NLP approaches including zero-shot learning and fine-tuned transformer models with PEFT (LoRA).

## 📁 Dataset

* **Source**: [Financial PhraseBank v1.0](https://www.researchgate.net/publication/251231364_FinancialPhraseBank-v10)
* **File**: `Sentences_75Agree.txt`
* **Task**: Classify financial sentences into:

  * `positive`
  * `neutral`
  * `negative`

## 🧹 Preprocessing

* Lowercasing, punctuation removal, and stopword filtering using NLTK.
* Mapped sentiment labels to integers.
* Cleaned and raw versions of text stored for modeling.

## 🧪 Models Implemented

### 1. **TF-IDF + Logistic Regression**

* Text vectorized with bi-gram TF-IDF (`max_features=10,000`).
* Hyperparameters tuned with Optuna.
* Used `OneVsRestClassifier` for multiclass support.

### 2. **TF-IDF + XGBoost**

* Same preprocessing as above.
* XGBoost classifier with `multi:softprob`.
* Hyperparameter tuning via Optuna.

### 3. **Zero-shot Classification with Transformers**

* Used `cardiffnlp/twitter-roberta-base-sentiment`.
* Inference via Hugging Face’s `TextClassificationPipeline`.
* No training required.

### 4. **Fine-tuned Transformer with LoRA (PEFT)**

* Base model: `cardiffnlp/twitter-roberta-base-sentiment`
* Applied parameter-efficient fine-tuning using:

  * `LoRAConfig`
  * `prepare_model_for_kbit_training`
* Trained via Hugging Face's `Trainer` API.
* Inference and error analysis with `transformers-interpret`.

## 📈 Metrics

Each model was evaluated using the following:

* Accuracy
* Precision (Macro)
* Recall (Macro)
* F1 Score (Weighted)
* ROC-AUC (One-vs-Rest)
* Average Precision Score

## 📊 Results Summary

| Model                       | Accuracy | Precision | Recall | F1 Score | ROC AUC | Avg. Precision |
|----------------------------|----------|-----------|--------|----------|---------|----------------|
| TF-IDF + Logistic Regression | 0.8408   | 0.8186    | 0.7310 | 0.8330   | 0.9259  | 0.8539         |
| TF-IDF + XGBoost            | 0.8133   | 0.7871    | 0.7062 | 0.8048   | 0.9054  | 0.8180         |
| Zero-shot LLM               | 0.6874   | 0.7025    | 0.5147 | 0.6545   | 0.8978  | 0.7811         |
| Fine-tuned LLM              | **0.9450**   | **0.9266**    | **0.9445** | **0.9452**   | **0.9921**  | **0.9851**         |


## 🔍 Interpretability

* Used `transformers-interpret` to visualize word-level attributions.
* Sampled and analyzed misclassified examples.

## ⚙️ Tools & Libraries

* `scikit-learn`, `xgboost`, `optuna`, `transformers`, `datasets`, `peft`
* `NLTK`, `matplotlib`, `evaluate`, `transformers-interpret`

## 📝 How to Run

1. Clone the repository and download the [Financial PhraseBank dataset](https://www.researchgate.net/publication/251231364_FinancialPhraseBank-v10).
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Run the main notebook or script:

   ```bash
   jupyter notebook sentiment_analysis_finance.ipynb
   ```
