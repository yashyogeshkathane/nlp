
# Robustness and Uncertainty in BERT-based Named Entity Recognition

This project explores the robustness, confidence, and transferability of a Named Entity Recognition (NER) system across domains. We use a BERT-based architecture, enhanced with Conditional Random Fields (CRF) and Monte Carlo Dropout, to investigate how a model trained on clean, formal data (CoNLL-2003) performs on noisy, real-world text (WNUT-17).

Key research questions include: How does a pretrained NER model behave on out-of-domain data? Can we quantify uncertainty and difficulty at sentence and token levels? What do embedding visualizations and classifier scores tell us about domain shift? How can these insights be used to improve model performance or interpretability?

---

## 🗂️ Datasets

### 1. **CoNLL-2003**
- Domain: Reuters newswire
- Entities: PER, ORG, LOC, MISC
- Characteristics: Clean, structured, and formal language

### 2. **WNUT-17**
- Domain: User-generated content (e.g., Twitter)
- Entities: person, location, corporation, product, creative-work, group
- Characteristics: Noisy, emerging entities with slang and irregular grammar

---

## 🎯 Project Objectives

### 🔍 Evaluation

- Zero-shot test a pretrained BERT-based NER model on CoNLL-2003 and WNUT-17.
- Normalize entity labels to enable fair cross-domain comparison.

### 📊 Analysis

- Train a logistic regression model to separate WNUT and CoNLL based on sentence embeddings.
- Use model scores as difficulty estimates and correlate with NER F1 performance.

### 🧪 Uncertainty

- Add a CRF decoding layer to model structured dependencies.
- Use Monte Carlo (MC) Dropout to estimate prediction variance.
- Visualize model uncertainty at token and sentence level.

---

## 🛠️ Methodology

### 1️⃣ Tokenization & Preprocessing

- Sentences are tokenized using BERT’s WordPiece tokenizer.
- Original labels are aligned with tokenized output.

### 2️⃣ Inference Pipeline

- Inference uses `dslim/bert-base-NER` without fine-tuning.
- Predictions are evaluated on both test sets using F1, Precision, Recall.

### 3️⃣ Difficulty Modeling with Logistic Regression

A logistic regression model is trained to distinguish between CoNLL and WNUT sentences based on the `[CLS]` token embeddings from BERT. The probability score produced by this model serves as a "WNUT-likeness" measure.

**📊 Score Distribution Plot:**  
This histogram displays the classifier’s score for each WNUT test sentence. Higher values indicate strong WNUT domain characteristics, while lower scores reflect CoNLL-like features. Most WNUT examples fall near a score of 1, suggesting strong domain distinction. [View Graph](visualization/output1.png)

### 4️⃣ t-SNE Visualization

**🌐 t-SNE Plot of Sentence Embeddings:**  
A 2D scatter plot is created using t-SNE on BERT embeddings. CoNLL and WNUT sentences are color-coded. The clusters are largely non-overlapping, demonstrating that BERT captures distinct domain-specific patterns in its latent space. [View Graph](visualization/output2.png)

---

## 🔍 Uncertainty Estimation: BERT + CRF + MC Dropout

### 🧩 Architecture

- Base Model: `bert-base-cased`
- Decoder: Linear → CRF Layer

### 🔄 Monte Carlo Dropout

Dropout layers are kept active during inference to simulate multiple forward passes. This allows estimation of variance across predictions, which is interpreted as uncertainty.

### 📊 Visualizations

**1. Sentence-Level Average Token Uncertainty:**  
This line plot shows the average standard deviation of token predictions for each sentence. Higher values indicate greater uncertainty and suggest ambiguous or difficult-to-parse content. This helps identify which sentences the model finds challenging. [View Graph](visualization/output3.png)

**2. Token-Level Uncertainty Heatmap:**  
This heatmap provides a token-by-token view of uncertainty for each sentence. Bright spots highlight tokens with high variance across MC Dropout iterations, often corresponding to rare, ambiguous, or out-of-vocabulary tokens. This can be used to debug and refine model predictions. [View Graph](visualization/output4.png)

**3. Distribution of Token-Level Prediction Uncertainty:**  
A histogram summarizing standard deviation values across all tokens. Most values lie between 2.0 and 3.0, with a few outliers at higher uncertainty. This gives a global view of prediction confidence across the dataset. [View Graph](visualization/output5.png)

---

## 🔍 Insights

- The BERT model shows **strong in-domain performance** (CoNLL), but **struggles with out-of-domain (WNUT)** text.
- Sentence-level embeddings allow us to analyze domain shift and classifier confidence.
- Token-level uncertainty via MC Dropout adds interpretability, showing where the model is unsure.
- Visualizations confirm the correlation between uncertainty and poor model predictions.

---

## 💡 Why This Matters

In real-world applications, NER systems often face **domain drift**—where the training data differs significantly from the deployment environment. This project:

- Demonstrates the **impact of domain shift** using real datasets.
- Introduces **uncertainty quantification** for NER models.
- Uses modern deep learning (BERT + CRF, MC Dropout) to probe model behavior **beyond accuracy metrics**.

---

## 🛠️ Setup

### Python Version
`Python 3.10`

---

## 👤 Author

**Your Name**  
[GitHub](https://github.com/yourusername) | [LinkedIn](https://linkedin.com/in/yourprofile)

---


