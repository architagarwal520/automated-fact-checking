# Introduction
As climate change becomes a more pressing issue, the public is increasingly exposed to conflicting and often unverified claims regarding climate science. Misinformation in this domain has contributed to confusion and skewed public opinion, highlighting the need for reliable fact-checking systems that can assess the accuracy of climate-related claims. Fact-checking plays a vital role in maintaining the integrity of public discourse and aiding informed decision-making, particularly in areas where scientific understanding is crucial, such as climate change.

This project aims to develop an Automated Fact-Checking System that classifies climate-related claims based on relevant evidence. Given a claim, the system retrieves the most pertinent evidence from a large corpus and classifies the claim into one of four categories: SUPPORTS, REFUTES, NOT_ENOUGH_INFO, or DISPUTED.

# Dataset Overview
The dataset for this project includes labeled claims from training and development sets (train-claims.json, dev-claims.json), and a corpus of evidence passages (evidence.json) that serve as the knowledge source. The claim-evidence relationship is mapped, where each claim is associated with relevant evidence passages. For evaluation, an unlabelled test set (test-claims-unlabelled.json) is used in conjunction with a Python script (eval.py) to measure system performance.

Each claim contains:
- Claim Text: The actual statement to be fact-checked.
- Claim Label: One of the four classification categories (SUPPORTS, REFUTES, etc.).
- Evidence IDs: References to evidence passages that correspond to the claim.

# Importance of Automated Fact-Checking
Given the scale of available data and the speed at which misinformation spreads, traditional manual fact-checking methods are insufficient to meet the demand for accurate information. Automating the fact-checking process using Natural Language Processing (NLP) techniques allows for faster, more reliable, and scalable solutions. This system can be adapted for other domains beyond climate science, improving the veracity of public information in various fields.

# Table of Contents

- Features
- Technology Stack
- Installation
- Usage
- Model Architecture
- Future Improvements

# Features
- Tokenization: Converts claims and evidence into tokens and prepares them for model input.
- LSA for Evidence Retrieval: Efficiently retrieves the most relevant evidence based on cosine similarity and semantic matching.
- LSTM-based Classification: Classifies claims using a Long Short-Term Memory (LSTM) network enhanced with an attention mechanism.
- Attention Mechanism: Highlights the most relevant parts of the evidence for better claim classification.

# Technology Stack

- Python
- Torch (PyTorch) for building and training the LSTM model
- scikit-learn for vectorization and LSA
- nltk for natural language preprocessing
- TensorFlow for tensor processing

# Installation

1. Clone the Repository  
   ```bash
      git clone https://github.com/yourusername/automated-fact-checking.git
      cd automated-fact-checking
   
2. Set up a Python Virtual Environment (Optional but Recommended):
   ```bash
      python3 -m venv venv
      source venv/bin/activate   # On Windows: venv\Scripts\activate
   
3. Install Required Dependencies:
   ```bash
      pip install -r requirements.txt
   
4. (Optional) Set Up for Google Colab:
If you are running this in Google Colab, ensure you have enabled GPU by navigating to Runtime > Change runtime type and selecting GPU.

5. Run the Jupyter Notebook:
   ```bash
   jupyter notebook

# Usage

1. Preprocess the data:
 - Tokenize claims and evidence.
 - Create a vocabulary from the tokenized data.
 - Convert claims and evidence to tensors.
2. Evidence Retrieval:
 - Use Latent Semantic Analysis (LSA) to match claims with the most relevant evidence based on cosine similarity.
3. LSTM-based Claim Classification:
 - Classify each claim into one of the four categories based on the evidence retrieved

# Model Architecture

The Automated Fact-Checking System uses a sophisticated deep learning architecture based on Long Short-Term Memory (LSTM) networks combined with a Word-Level Attention Mechanism. This architecture is well-suited for handling sequential textual data and allows the model to efficiently process claims and evidence for fact-checking.

Key Components of the Model

1. Embedding Layer

  The embedding layer is the first step of the model and is used to convert tokenized claims and evidence into dense vector representations. Each token (word) in the claim and evidence is mapped to a continuous-valued   vector, where semantically similar words are placed close together in the vector space.
  
  Why Embeddings?: Word embeddings capture the semantic meaning of words and reduce the dimensionality of the input text, making it easier for the model to learn patterns across large vocabulary sizes.

2. LSTM Layers

  The model uses two separate LSTM (Long Short-Term Memory) networks:
  
  One LSTM processes the claims.
  Another LSTM processes the evidence.
  The LSTM architecture is chosen because of its ability to handle long-term dependencies in textual data. LSTM networks are capable of retaining information over long sequences, which is critical for understanding      complex relationships between claims and evidence.
  
  Why LSTM?: Climate-related claims and evidence often involve long sentences with intricate dependencies. LSTMs help retain important contextual information over long input sequences, improving the model’s ability to   understand and relate claims to evidence.
  
3. Word-Level Attention Mechanism

  The attention mechanism is applied at the word level to calculate the relevance of each word in the evidence with respect to the claim. The attention mechanism assigns higher weights to the words that are most         relevant to the claim, allowing the model to focus on the critical parts of the evidence.
  
  Why Word-Level Attention?
  Interpretability: Attention helps the model highlight which words or phrases in the evidence are most important for making decisions, making the model’s predictions more interpretable.
  Efficiency: Instead of treating every word in the evidence equally, the attention mechanism enables the model to focus only on the relevant parts of the evidence. This reduces the noise introduced by irrelevant        words, improving classification accuracy.
  Better Representation: By dynamically weighting the importance of different words, the model can create a more meaningful representation of the evidence that is tailored to the specific claim being evaluated.
  The attention mechanism works as follows:
  
  The LSTM outputs for the claim and evidence are fed into an attention layer.
  The model computes a compatibility score between the claim and each word in the evidence by performing a matrix multiplication between the LSTM outputs of the claim and evidence.
  These scores are passed through a softmax function to generate normalized attention weights.
  The attention weights are used to create a context vector by taking a weighted sum of the evidence LSTM outputs.
  The context vector is then combined with the claim’s LSTM output to form the final representation for classification.

4. Final LSTM Layer

  After the attention mechanism, the final LSTM layer takes the combined representation of the claim and evidence and processes it to capture any remaining dependencies. This allows the model to refine the contextual    relationship between the claim and the evidence further.

5. Fully Connected Layer (FFNN)
  
  The final LSTM output is passed to a Fully Connected Layer (FFNN) that performs the classification. The FFNN maps the final output to one of the four possible classes:
  
  SUPPORTS: The evidence supports the claim.
  REFUTES: The evidence refutes the claim.
  NOT_ENOUGH_INFO: The evidence does not provide enough information to verify the claim.
  DISPUTED: The evidence presents conflicting information regarding the claim.
  The classification is done using a softmax activation function, which outputs probabilities for each class, and the class with the highest probability is chosen as the final prediction.


# Model Training

Model Parameters:
  - Vocabulary Size: The number of unique tokens in the data.
  - Embedding Dimension: 50, which is the size of the vector space where each word is mapped.
  - Hidden Dimension: 128, which defines the size of the hidden state in the LSTM layers.
  - Number of Layers: 2 LSTM layers, providing the model with the capacity to capture complex patterns in the data.
  - Dropout: 0.5, which is applied to prevent overfitting by randomly turning off neurons during training.

Training Loss:
  - The training loss decreases steadily over the 10 epochs, starting from around 1.28 and ending at around 1.14.
  - This indicates that the model is learning and improving its performance on the training data. The model is adjusting its weights effectively during backpropagation to minimize the loss.
  - A continuously decreasing training loss is a positive sign that the model is able to fit the training data well.

# Model Evaluation

Training Loss:

 - The training loss continuously decreases over the 10 epochs, which indicates that the model is learning the patterns in the training data. This is a typical behavior as the model minimizes the loss function by       updating weights through backpropagation.
 - A decreasing training loss is usually a good sign that the model is fitting the data well. However, it is important to monitor both training and validation loss/accuracy to ensure that the model is not overfitting.
 - 
Validation Loss and Accuracy:

 - The validation loss starts increasing after the initial few epochs. This is a clear indicator of overfitting. Overfitting happens when the model is learning the training data too well, capturing noise and specific   patterns that do not generalize to unseen data (i.e., the validation set).
 - The validation accuracy is unstable and starts decreasing after a few epochs, which is another signal of overfitting. While the model is performing well on the training set (as seen by the decreasing training        loss), it is not generalizing to the validation set.

