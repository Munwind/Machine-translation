# Transformer-based Bilingual Translation Model

This repository implements a transformer-based model for bilingual translation. The model follows the original transformer architecture (with encoder–decoder structure) and is built using PyTorch. It includes custom modules for input embeddings, positional encoding, multi-head attention, feed-forward layers, layer normalization, residual connections, and both encoder and decoder blocks. In addition, the code includes utilities for building tokenizers with the [tokenizers](https://github.com/huggingface/tokenizers) library and data loading using [Hugging Face Datasets](https://huggingface.co/datasets).

## Overview

The project implements a full end-to-end pipeline for training a translation model:
- **Data Preparation:** Uses the `opus_books` dataset to extract parallel sentences. Custom tokenizers are built for both source and target languages.
- **Model Architecture:** A transformer with separate encoder and decoder stacks, along with input embedding and positional encoding layers. A projection layer maps the decoder’s output to the target vocabulary.
- **Training and Evaluation:** Includes a training loop with teacher forcing, validation routines using greedy decoding, and TensorBoard logging.
- **Inference:** A greedy decoding function is provided to sample translations for a given source sentence.

## Model Architecture

The model is built from several key components:

- **InputEmbeddings:** Scales word embeddings by the square root of the model dimension.
- **PositionalEncoding:** Adds sine and cosine positional encodings to token embeddings.
- **LayerNormalization:** Normalizes inputs across the feature dimension.
- **FeedForward:** A two-layer feed-forward network with ReLU activation.
- **MultiheadAttention:** Computes scaled dot-product attention across multiple heads.
- **ResidualConnection:** Wraps a sublayer with dropout and a residual connection.
- **EncoderBlock & Encoder:** Stacks self-attention and feed-forward layers to build the encoder.
- **DecoderBlock & Decoder:** Uses self-attention, cross-attention (to the encoder output), and feed-forward layers to build the decoder.
- **ProjectionLayer:** Projects decoder outputs to log-probabilities over the target vocabulary.
- **Transformers:** Combines the above components into a full transformer model.

The function `build_transformers` creates an instance of the transformer model given vocabulary sizes, sequence lengths, and other hyperparameters.

## Installation

Ensure you have Python 3.7+ and install the required packages. You can install the dependencies with:

```bash
pip install torch torchvision datasets tokenizers tqdm tensorboard
