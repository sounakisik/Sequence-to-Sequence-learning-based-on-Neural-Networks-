# Sequence to Sequence Learning using LSTM and RNN

This project implements a basic Sequence-to-Sequence learning framework using RNN/LSTM encoder-decoder architecture. The model is trained for English-to-Hindi translation, where the encoder reads an input English sentence and compresses it into a hidden representation, while the decoder generates the corresponding Hindi sentence token by token.

The project demonstrates the core idea behind neural machine translation using recurrent architectures, including vocabulary building, tokenization, padding, teacher forcing, loss minimization, and greedy decoding.

## Project Objective

The main objective of this project is to build a neural network that can learn a mapping from one variable-length sequence to another variable-length sequence.

In this implementation:

- Source sequence: English sentence
- Target sequence: Hindi sentence
- Model: Encoder-Decoder using RNN or LSTM
- Training method: Teacher forcing
- Loss function: Cross-Entropy Loss
- Decoding method: Greedy decoding

## Methodology

The project follows the below steps:

1. Load English-Hindi sentence pairs.
2. Clean and tokenize the sentences.
3. Add special tokens:
   - `<SOS>`: Start of sentence
   - `<EOS>`: End of sentence
   - `<PAD>`: Padding token
   - `<UNK>`: Unknown token
4. Build source and target vocabularies.
5. Convert sentences into integer token sequences.
6. Train an Encoder-Decoder model using RNN/LSTM.
7. Generate Hindi translations for new English input sentences.

## Model Architecture

### Encoder

The encoder receives the English sentence token by token. Each token is first converted into an embedding vector. The embedded sequence is then passed through an RNN/LSTM layer. The final hidden state of the encoder acts as a compressed representation of the source sentence.

### Decoder

The decoder uses the encoder hidden state to generate the target Hindi sentence. During training, the decoder receives the actual previous target word using teacher forcing. During inference, the decoder uses its own previously predicted word to generate the next word.

## Technologies Used

- Python
- PyTorch
- NumPy
- Pandas

## Dataset Format

The code expects a CSV file named:

```text
data/eng_hindi.csv
