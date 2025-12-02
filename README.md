# News Topic Classification with a Feedforward Neural Network

This project implements a **from-scratch feedforward neural network** (no deep learning frameworks) to classify news articles into three topics:

- **Politics**
- **Sports**
- **Economy**

It was developed as part of the **COM4513/6513 – Natural Language Processing** assignment at the University of Sheffield.

The full solution is implemented in a Jupyter notebook, including:
- Custom text preprocessing
- Manual implementation of forward and backward passes
- Stochastic Gradient Descent (SGD) training with dropout
- Experiments with randomly initialised embeddings and pre-trained GloVe embeddings
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
- **text** – the raw news article content

In the code, labels are shifted to start from 0 (`0, 1, 2`) and one-hot encoded for training.

---

## Text Preprocessing

The preprocessing pipeline is implemented with NumPy and regular expressions:

1. **Tokenisation**
   - Text is lowercased.
   - Tokens are extracted using a regex pattern that keeps alphabetic tokens only.

2. **Stopword Removal**
   - A custom stopword list is applied (e.g. `a, in, on, at, and, or, to, the, of, ...`).

3. **Vocabulary Construction**
   - Unigrams are extracted from the training and development sets.
   - Terms appearing in fewer than a minimum number of documents are filtered out.
   - Only the top 3,000 most frequent unigrams are kept to control dimensionality and reduce noise.

4. **Indexing**
   - Two dictionaries are created:
     - `vocab2id`: word → index  
     - `id2vocab`: index → word
   - Each document is represented as a list of vocabulary indices instead of sparse one-hot vectors.

5. **Label Encoding**
   - Original labels `{1, 2, 3}` are mapped to `{0, 1, 2}` by subtracting 1.
   - Labels are converted to one-hot vectors.

---

## Model Architecture

The core model is a **feedforward neural network** with the following structure:

1. **Embedding Layer**
   - Vocabulary size ≈ 3,000
   - Embedding dimension: 300
   - Each document is represented by the mean of its word embeddings:
     \[
     h_1 = \frac{1}{|x|} \sum_{i \in x} W_e[i]
     \]
   - Implemented as an embedding matrix `W[0]` indexed by word IDs.

2. **Hidden Layer(s)**
   - Baseline model:
     - One hidden layer with 128 ReLU units.
   - Extended model:
     - Additional hidden layer(s) stacked on top of the first hidden layer.

   Activation function: ReLU
   \[
   \mathrm{ReLU}(z) = \max(0, z)
   \]

3. **Dropout**
   - Dropout is applied after each hidden layer with dropout rate 0.2.
   - Implemented via a binary mask sampled from a Bernoulli distribution and applied elementwise.

4. **Output Layer**
   - Fully connected layer mapping to 3 classes.
   - Softmax activation to obtain class probabilities.

5. **Loss**
   - Categorical cross-entropy:
     \[
     \mathcal{L}(y, \hat{y}) = -\sum_{c} y_c \log \hat{y}_c
     \]
   - Implemented with numerical stability (clipping + small epsilon inside the log).

6. **Optimisation**
   - Stochastic Gradient Descent (SGD) with instance-wise updates.
   - Manual backpropagation is implemented for all layers, including the embedding layer (unless explicitly frozen).

---

## Experiments

### 1. Baseline: Randomly Initialised Embeddings

- **Architecture**
  - 300-dimensional embeddings
  - Hidden layer: 128 units, ReLU
  - Dropout: 0.2

- **Training**
  - Learning rate: 0.005
  - Up to 100 epochs with early stopping based on validation loss

- **Test performance**
  - Accuracy: **0.84**
  - Precision (macro): **0.84**
  - Recall (macro): **0.84**
  - F1-Score (macro): **0.84**

This model achieves strong and balanced performance across all metrics and serves as the main reference.

---

### 2. Pre-trained GloVe Embeddings (Frozen)

- 300-dimensional GloVe Common Crawl embeddings are loaded from `glove.840B.300d.txt`.
- The embedding matrix is initialised from GloVe and **frozen** during training.
- Same architecture and training setup as the baseline.

- **Test performance (frozen GloVe)**
  - Accuracy: **0.73**
  - Precision (macro): **0.73**
  - Recall (macro): **0.73**
  - F1-Score (macro): **0.73**

Despite using richer pre-trained embeddings, freezing them leads to lower performance than the baseline, likely because the embeddings are not fine-tuned to this specific dataset.

---

### 3. GloVe + Deeper Network

- Pre-trained GloVe embeddings (frozen)
- Extended architecture with additional hidden layer(s) on top of the average embedding

- **Test performance (GloVe + deeper network)**
  - Accuracy: **0.76**
  - Precision (macro): **0.85**
  - Recall (macro): **0.78**
  - F1-Score (macro): **0.75**

This setup improves precision and recall compared to the simple GloVe model, but it still does not surpass the baseline with randomly initialised embeddings.

---

## Learning Curves & Training Behaviour

- Training and validation loss curves are plotted over epochs.
- For the baseline model:
  - Training loss decreases steadily.
  - Validation loss initially decreases and then slowly rises.
  - Early stopping is triggered once the validation loss change falls below a small tolerance, preventing overfitting and unnecessary computation.

---

## Error Analysis (High-Level)

The error analysis in the notebook highlights several patterns:

- **False positives**  
  Some articles are predicted as a given class (e.g. sports) when they actually belong to another (e.g. politics), usually when they share overlapping vocabulary.

- **False negatives**  
  Some true class instances are missed, suggesting recall could be further improved.

- **Class confusion**  
  Misclassifications often happen between semantically related topics or when the article is short and lacks clear topic-specific keywords.

Overall, the model generalises well but could be further improved with more sophisticated architectures or additional features.

---
