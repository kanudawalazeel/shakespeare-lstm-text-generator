# Shakespearean Text Generator using LSTM

A character-level text generation model built with TensorFlow/Keras, trained on Shakespeare's complete works. This project covers the full pipeline — data preprocessing, model design, training, and text generation — along with architecture experiments and performance evaluation.

## Overview

- **Type:** Character-level language model
- **Architecture:** Embedding layer → Stacked LSTM layers → Dense (softmax) output
- **Loss:** Sparse categorical crossentropy
- **Optimizer:** Adam
- **Regularization:** Dropout, Early Stopping, Model Checkpointing

## Dataset

Shakespeare's Complete Works (public domain).
Source: [Shakespeare Plays - Kaggle](https://www.kaggle.com/datasets/kewagbln/shakespeareonline)

## Pipeline

1. **Preprocessing** — Lowercased text, removed unwanted characters, character-level tokenization.
2. **Sequence creation** — Sliding window of 100 characters as input, next character as target.
3. **Model** — Embedding layer + 2 stacked LSTM layers (256 units each) + Dropout + Dense softmax output.
4. **Training** — Trained with early stopping and checkpointing on train/validation split.
5. **Text generation** — Iterative next-character prediction with temperature-controlled sampling.

## Bonus: Architecture Experiments

A deeper 3-layer LSTM variant was trained and compared against the baseline using validation loss, accuracy, and perplexity.

## Performance Metrics

| Model              | Val Loss | Val Accuracy | Perplexity |
|--------------------|----------|--------------|------------|
| Baseline (2-LSTM)  | (see notebook output) | | |
| Deeper (3-LSTM)    | (see notebook output) | | |

## Sample Output

Generated text examples from multiple seed phrases (e.g., "to be or not to be", "all the world's a stage") are included in the notebook, along with comparisons across temperature settings and model architectures.

## How to Run

1. Clone this repo
2. Install dependencies:
pip install -r requirements.txt
3. Download the dataset "Shakespeare Plays" - Kaggle and place it in the appropriate path
4. Open and run `shakespeare_lstm_text_generator.ipynb` in Jupyter, Kaggle, or Google Colab

## Tech Stack

Python, TensorFlow/Keras, NumPy, Matplotlib, scikit-learn

## Notes

This notebook was developed and trained on Kaggle with GPU acceleration. Saved model weights and tokenizer mappings are available in the Kaggle notebook's output for reuse without retraining.
