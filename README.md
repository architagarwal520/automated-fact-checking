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
