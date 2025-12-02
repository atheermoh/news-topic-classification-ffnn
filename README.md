# News Topic Classification using a Feedforward Neural Network (NumPy Only)

This project implements a **Feedforward Neural Network (FFNN)** for multi-class news topic classification (Politics, Sports, Economy).  
The entire neural network — embeddings, hidden layers, forward pass, backpropagation, dropout, softmax, and optimisation — is implemented **manually in NumPy**.

## Dataset

The project uses a subset of the **AG News corpus**:

- `data_topic/train.csv` – 2,400 articles (800 per class)
- `data_topic/dev.csv` – 150 articles (50 per class)
- `data_topic/test.csv` – 900 articles (300 per class)

Each instance consists of:

- **label** – integer in `{1, 2, 3}` corresponding to:
  - 1 → Politics  
  - 2 → Sports  
  - 3 → Economy
- **text** – the raw news article content

In the code, labels are shifted to start from 0 (`0, 1, 2`) and one-hot encoded for training.

---
## Problem Definition

Given an input text (news article), the goal is to predict one of three classes:

- `0`: Politics  
- `1`: Sports  
- `2`: Economy  

This is a **multi-class classification** problem with a **probability distribution output** using softmax.

---

## Model Architecture

### **1. Embedding Layer**
- Vocabulary size ≈ 3,000 unigrams
- Each token ID is mapped to a 300-dimensional embedding vector
- Document embedding = **mean of all token embeddings**

Mathematically:
h0 = mean( W_emb[ token_indices ] )

where:
- `W_emb` is the embedding matrix (V × 300)
- `token_indices` are the vocabulary IDs for the document

Two modes supported:
- **Random embeddings (learnable)**
- **GloVe embeddings (frozen or learnable)**

---

### **2. Hidden Layer (ReLU)**
A single fully-connected hidden layer:

h1 = ReLU( h0 ⋅ W1 )

where:
- `W1` is (300 × 128)
- ReLU(x) = max(x, 0)

---

### **3. Dropout (Training Only)**

A dropout mask `m` sampled from Bernoulli(1 − dropout_rate):

h1_dropout = h1 * m

---

### **4. Output Layer + Softmax**

logits = h1_dropout ⋅ W2
ŷ = softmax(logits)

Softmax implementation (numerically stable):

softmax(z) = exp(z - max(z)) / sum( exp(z - max(z)) )

---

### **5. Loss Function: Categorical Cross-Entropy**
L(y, ŷ) = - sum_over_classes( y_c * log(ŷ_c) )

Where:
- `y` is the one-hot true label vector
- `ŷ` is the predicted probability distribution

Only the log-probability of the true class contributes to the loss.

---

### **6. Optimisation: Stochastic Gradient Descent (SGD)**

SGD is performed **instance-by-instance**:

W ← W - lr * ∇W(L)

Backpropagation is implemented manually:

- gradient of softmax + cross-entropy  
- gradient through dropout  
- gradient through ReLU  
- gradient through fully-connected layers  
- optional: gradient blocked for frozen embeddings (GloVe)

---

## Text Preprocessing Pipeline
The preprocessing pipeline is implemented with NumPy and regular expressions:

1. **Regex tokenisation**  
   Extract alphabetic tokens (`[A-Za-z]+`), convert to lowercase.

2. **Stopword removal**

3. **Document frequency filtering**  
   Remove tokens appearing in fewer than 2 documents.

4. **Top-N pruning**  
   Keep the top 3,000 unigrams by frequency.

5. **Vocabulary mapping**  
   - `vocab2id`: token → integer ID  
   - `id2vocab`: integer ID → token

6. **Document to index conversion**  
   Every document becomes a list of token IDs.

7. **One-hot encoding for labels**  
   For 3 classes:
y = [1,0,0], [0,1,0], or [0,0,1]

---

##  Model Performance

### **Baseline (Random Initial Embeddings)**
- Accuracy: **0.84**
- Precision (macro): **0.84**
- Recall (macro): **0.84**
- F1-score (macro): **0.84**

This is the strongest model.

---

### **GloVe Embeddings (Frozen)**
- Accuracy: **0.73**

Freezing embeddings limits adaptability to this dataset.

---

### **GloVe + Additional Hidden Layer**
- Accuracy: **0.76**
- Precision (macro): **0.85**
- F1-score (macro): **0.75**

Adding depth improves expressiveness but still underperforms the baseline.

---

## Training Behaviour

- **Training loss decreases smoothly**
- **Validation loss decreases, then rises slightly**
- Early stopping triggers based on a tolerance threshold to prevent overfitting
- Loss curves help visualise convergence and generalisation

Figures provided:
- baseline loss curve
- GloVe loss curve
- metrics bar plot



