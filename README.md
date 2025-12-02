# news-topic-classification-ffnn
Feedforward neural network for classifying news articles into Politics, Sports, and Economy using custom NumPy implementation, SGD, dropout, and GloVe embeddings.

# News Topic Classification with a Feedforward Neural Network

This project implements a **from-scratch feedforward neural network** (no deep learning frameworks) to classify news articles into three topics:

- **Politics**
- **Sports**
- **Economy**

It was developed as part of the **COM4513/6513 – Natural Language Processing** assignment at the University of Sheffield.:contentReference[oaicite:0]{index=0}

The full solution is implemented in a Jupyter notebook, including:
- Custom text preprocessing
- Manual implementation of forward and backward passes
- Stochastic Gradient Descent (SGD) training with dropout
- Experiments with **randomly initialised embeddings** and **pre-trained GloVe embeddings**
- Extension with a deeper network architecture

---

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
- **text** – the raw news article text:contentReference[oaicite:1]{index=1}

In the code, labels are shifted to start from 0 (`0, 1, 2`) and one-hot encoded for training.:contentReference[oaicite:2]{index=2}

---

## Text Preprocessing

The preprocessing pipeline is fully implemented with **NumPy** and **regular expressions**:

1. **Tokenisation**
   - Tokens extracted using a regex pattern (alphabetic tokens only).
   - Text is lowercased.

2. **Stopword Removal**
   - A custom stopword list is applied (e.g. `a, in, on, at, and, or, ...`).:contentReference[oaicite:3]{index=3}

3. **Vocabulary Construction**
   - Unigrams are extracted from **train + dev** sets.
   - Terms appearing in fewer than a minimum number of documents are filtered out.
   - Only the **top 3,000 most frequent unigrams** are kept to control dimensionality and reduce noise.:contentReference[oaicite:4]{index=4}

4. **Indexing**
   - Two dictionaries are created:
     - `vocab2id`: word → index  
     - `id2vocab`: index → word:contentReference[oaicite:5]{index=5}
   - Each document is represented as a **list of vocabulary indices** instead of sparse one-hot vectors.

5. **Label Encoding**
   - Original labels `{1, 2, 3}` → `{0, 1, 2}` by subtracting 1.
   - Labels are converted to **one-hot vectors** using an identity matrix.:contentReference[oaicite:6]{index=6}

---

## Model Architecture

The core model is a **feedforward neural network** with the following structure:

1. **Embedding Layer**
   - Vocabulary size ≈ 3,000
   - Embedding dimension: **300**
   - Each document is represented by the **mean of its word embeddings**:
     \[
     h_1 = \frac{1}{|x|} \sum_{i \in x} W_e[i]
     \]
   - This is implemented as an embedding matrix `W[0]` indexed by word IDs.:contentReference[oaicite:7]{index=7}

2. **Hidden Layer(s)**
   - Baseline model:
     - One hidden layer with **128 ReLU units**.
   - Extended model:
     - Additional hidden layer(s) stacked on top of the first.

   Activation:
   - **ReLU**:
     \[
     \mathrm{ReLU}(z) = \max(0, z)
     \]

3. **Dropout**
   - Dropout is applied **after each hidden layer** with rate **0.2**.
   - Implemented via a binary mask sampled from `Bernoulli(1 - dropout_rate)` and applied elementwise.:contentReference[oaicite:9]{index=9}

4. **Output Layer**
   - Fully connected layer mapping to **3 classes**.
   - **Softmax** activation to obtain class probabilities.

5. **Loss**
   - **Categorical cross-entropy** is used:
     \[
     \mathcal{L}(y, \hat{y}) = -\sum_{c} y_c \log \hat{y}_c
     \]
   - Implemented with numerical stability (clipping + log-epsilon).:contentReference[oaicite:10]{index=10}

6. **Optimisation**
   - **Stochastic Gradient Descent (SGD)** with instance-wise updates.
   - Manual backpropagation is implemented for all layers, including the embedding layer (unless frozen).:contentReference[oaicite:11]{index=11}

---

## Experiments

### 1. Baseline: Randomly Initialised Embeddings

- **Architecture**:  
  - 300-d embeddings  
  - Hidden layer: 128 units, ReLU  
  - Dropout: 0.2  
- **Training**:
  - Learning rate: `0.005`  
  - Epochs: up to 100 with **early stopping** based on dev loss.
- **Result on test set**:
  - Accuracy: **0.84**
  - Precision (macro): **0.84**
  - Recall (macro): **0.84**
  - F1-Score (macro): **0.84**:contentReference[oaicite:13]{index=13}

This model shows a good balance between precision and recall and serves as the main reference.

---

### 2. Pre-trained GloVe Embeddings (Frozen)

- 300-d **GloVe Common Crawl** embeddings are loaded from `glove.840B.300d.txt`.:contentReference[oaicite:14]{index=14}
- The embedding matrix `W[0]` is **initialised from GloVe** and **frozen** during training (no gradient updates).
- Same architecture and training setup as the baseline.

**Result on test set** (from final summary table in the notebook):

- Accuracy: **0.73**
- Precision (macro): **0.73**
- Recall (macro): **0.73**
- F1-Score (macro): **0.73**:contentReference[oaicite:15]{index=15}

Despite using richer embeddings, this setup underperforms the baseline, likely due to a mismatch between the frozen embeddings and the small, domain-specific dataset.

---

### 3. GloVe + Deeper Network

- Pre-trained GloVe embeddings (frozen)  
- Extended architecture with **additional hidden layer(s)** on top of the average embedding.:contentReference[oaicite:16]{index=16}

**Result on test set** (summary table):

- Accuracy: **0.76**
- Precision (macro): **0.85**
- Recall (macro): **0.78**
- F1-Score (macro): **0.75**:contentReference[oaicite:17]{index=17}

This model gains precision but loses overall accuracy compared to the baseline. It performs better than simple GloVe with a single layer, but still does not surpass the randomly initialised embedding model.

---

## Learning Curves & Training Behaviour

- Training and validation loss curves are plotted over epochs.
- For the baseline model with random embeddings:
  - Train loss decreases steadily.
  - Validation loss initially decreases and then begins to rise slightly.
  - **Early stopping** is triggered once the validation loss stops improving beyond a small tolerance, preventing overfitting.

These curves provide evidence that the model is trained to a good point without excessive overfitting.

---

## Error Analysis (High-level)

From the final analysis in the notebook:​:contentReference[oaicite:19]{index=19}

- **False Positives**  
  Some articles are predicted as a given class when they actually belong to another (e.g., sports vs. politics). Thresholding or better regularisation could help.

- **False Negatives**  
  The model occasionally misses true positives, suggesting recall can still be improved.

- **Confusion Between Classes**  
  Misclassifications often occur between semantically similar topics or when the article uses ambiguous vocabulary.

- **Potential Class Imbalance**  
  While the dataset is fairly balanced, further work could explore class weighting or resampling techniques to improve robustness.

---


