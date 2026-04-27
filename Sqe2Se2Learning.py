
## `seq2seq_lstm_rnn.py`

```python
import re
import random
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple


# -----------------------------
# 1. Configuration
# -----------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = "data/eng_hindi.csv"

MIN_FREQ = 1
MAX_LEN = 20
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 1
DROPOUT = 0.2
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
TEACHER_FORCING_RATIO = 0.5

MODEL_TYPE = "lstm"  # choose "rnn" or "lstm"


# -----------------------------
# 2. Text Preprocessing
# -----------------------------

def clean_english_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"[^a-zA-Z?.!,]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_hindi_text(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    return text.split()


# -----------------------------
# 3. Vocabulary Class
# -----------------------------

class Vocabulary:
    def __init__(self):
        self.pad_token = "<PAD>"
        self.sos_token = "<SOS>"
        self.eos_token = "<EOS>"
        self.unk_token = "<UNK>"

        self.word2idx = {
            self.pad_token: 0,
            self.sos_token: 1,
            self.eos_token: 2,
            self.unk_token: 3,
        }

        self.idx2word = {
            0: self.pad_token,
            1: self.sos_token,
            2: self.eos_token,
            3: self.unk_token,
        }

        self.word_freq = {}

    def __len__(self):
        return len(self.word2idx)

    def build_vocab(self, sentences: List[str], min_freq: int = 1):
        for sentence in sentences:
            for word in tokenize(sentence):
                self.word_freq[word] = self.word_freq.get(word, 0) + 1

        for word, freq in self.word_freq.items():
            if freq >= min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def numericalize(self, sentence: str) -> List[int]:
        tokens = tokenize(sentence)
        return [
            self.word2idx.get(token, self.word2idx[self.unk_token])
            for token in tokens
        ]

    def decode_indices(self, indices: List[int]) -> str:
        words = []
        for idx in indices:
            word = self.idx2word.get(idx, self.unk_token)
            if word == self.eos_token:
                break
            if word not in [self.pad_token, self.sos_token]:
                words.append(word)
        return " ".join(words)


# -----------------------------
# 4. Dataset Preparation
# -----------------------------

def load_dataset(path: str) -> Tuple[List[str], List[str]]:
    df = pd.read_csv(path)

    if "english" not in df.columns or "hindi" not in df.columns:
        raise ValueError("CSV must contain two columns: 'english' and 'hindi'.")

    english_sentences = df["english"].apply(clean_english_text).tolist()
    hindi_sentences = df["hindi"].apply(clean_hindi_text).tolist()

    filtered_english = []
    filtered_hindi = []

    for eng, hin in zip(english_sentences, hindi_sentences):
        if len(tokenize(eng)) <= MAX_LEN and len(tokenize(hin)) <= MAX_LEN:
            filtered_english.append(eng)
            filtered_hindi.append(hin)

    return filtered_english, filtered_hindi


def sentence_to_tensor(sentence: str, vocab: Vocabulary) -> torch.Tensor:
    indices = [vocab.word2idx[vocab.sos_token]]
    indices += vocab.numericalize(sentence)
    indices += [vocab.word2idx[vocab.eos_token]]
    return torch.tensor(indices, dtype=torch.long)


def create_batches(
    source_sentences: List[str],
    target_sentences: List[str],
    source_vocab: Vocabulary,
    target_vocab: Vocabulary,
    batch_size: int
):
    data = list(zip(source_sentences, target_sentences))
    random.shuffle(data)

    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]

        source_tensors = [
            sentence_to_tensor(src, source_vocab)
            for src, _ in batch
        ]

        target_tensors = [
            sentence_to_tensor(trg, target_vocab)
            for _, trg in batch
        ]

        source_padded = nn.utils.rnn.pad_sequence(
            source_tensors,
            padding_value=source_vocab.word2idx[source_vocab.pad_token],
            batch_first=True
        )

        target_padded = nn.utils.rnn.pad_sequence(
            target_tensors,
            padding_value=target_vocab.word2idx[target_vocab.pad_token],
            batch_first=True
        )

        yield source_padded.to(DEVICE), target_padded.to(DEVICE)


# -----------------------------
# 5. Encoder
# -----------------------------

class Encoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        model_type: str = "lstm"
    ):
        super().__init__()

        self.model_type = model_type
        self.embedding = nn.Embedding(input_dim, embedding_dim)

        if model_type == "lstm":
            self.rnn = nn.LSTM(
                embedding_dim,
                hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        elif model_type == "rnn":
            self.rnn = nn.RNN(
                embedding_dim,
                hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        else:
            raise ValueError("model_type must be either 'rnn' or 'lstm'.")

        self.dropout = nn.Dropout(dropout)

    def forward(self, source):
        embedded = self.dropout(self.embedding(source))
        outputs, hidden = self.rnn(embedded)
        return outputs, hidden


# -----------------------------
# 6. Decoder
# -----------------------------

class Decoder(nn.Module):
    def __init__(
        self,
        output_dim: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        model_type: str = "lstm"
    ):
        super().__init__()

        self.output_dim = output_dim
        self.model_type = model_type

        self.embedding = nn.Embedding(output_dim, embedding_dim)

        if model_type == "lstm":
            self.rnn = nn.LSTM(
                embedding_dim,
                hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        elif model_type == "rnn":
            self.rnn = nn.RNN(
                embedding_dim,
                hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        else:
            raise ValueError("model_type must be either 'rnn' or 'lstm'.")

        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_token, hidden):
        input_token = input_token.unsqueeze(1)
        embedded = self.dropout(self.embedding(input_token))

        output, hidden = self.rnn(embedded, hidden)

        prediction = self.fc_out(output.squeeze(1))

        return prediction, hidden


# -----------------------------
# 7. Seq2Seq Model
# -----------------------------

class Seq2Seq(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        target_vocab_size: int,
        device
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.target_vocab_size = target_vocab_size
        self.device = device

    def forward(self, source, target, teacher_forcing_ratio=0.5):
        batch_size = source.shape[0]
        target_len = target.shape[1]

        outputs = torch.zeros(
            batch_size,
            target_len,
            self.target_vocab_size
        ).to(self.device)

        _, hidden = self.encoder(source)

        input_token = target[:, 0]

        for t in range(1, target_len):
            output, hidden = self.decoder(input_token, hidden)
            outputs[:, t, :] = output

            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)

            input_token = target[:, t] if teacher_force else top1

        return outputs


# -----------------------------
# 8. Training Function
# -----------------------------

def train_epoch(
    model,
    source_sentences,
    target_sentences,
    source_vocab,
    target_vocab,
    optimizer,
    criterion
):
    model.train()
    epoch_loss = 0
    batch_count = 0

    for source_batch, target_batch in create_batches(
        source_sentences,
        target_sentences,
        source_vocab,
        target_vocab,
        BATCH_SIZE
    ):
        optimizer.zero_grad()

        output = model(
            source_batch,
            target_batch,
            teacher_forcing_ratio=TEACHER_FORCING_RATIO
        )

        output_dim = output.shape[-1]

        output = output[:, 1:, :].reshape(-1, output_dim)
        target_batch = target_batch[:, 1:].reshape(-1)

        loss = criterion(output, target_batch)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        optimizer.step()

        epoch_loss += loss.item()
        batch_count += 1

    return epoch_loss / batch_count


# -----------------------------
# 9. Translation / Inference
# -----------------------------

def translate_sentence(
    sentence: str,
    model: Seq2Seq,
    source_vocab: Vocabulary,
    target_vocab: Vocabulary,
    max_len: int = 30
) -> str:
    model.eval()

    sentence = clean_english_text(sentence)

    source_tensor = sentence_to_tensor(sentence, source_vocab)
    source_tensor = source_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        _, hidden = model.encoder(source_tensor)

    input_token = torch.tensor(
        [target_vocab.word2idx[target_vocab.sos_token]],
        dtype=torch.long
    ).to(DEVICE)

    predicted_indices = []

    for _ in range(max_len):
        with torch.no_grad():
            output, hidden = model.decoder(input_token, hidden)

        predicted_token = output.argmax(1).item()

        if predicted_token == target_vocab.word2idx[target_vocab.eos_token]:
            break

        predicted_indices.append(predicted_token)

        input_token = torch.tensor([predicted_token], dtype=torch.long).to(DEVICE)

    return target_vocab.decode_indices(predicted_indices)


# -----------------------------
# 10. Save and Load Utilities
# -----------------------------

def save_checkpoint(model, source_vocab, target_vocab, path="seq2seq_model.pth"):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "source_word2idx": source_vocab.word2idx,
        "source_idx2word": source_vocab.idx2word,
        "target_word2idx": target_vocab.word2idx,
        "target_idx2word": target_vocab.idx2word,
    }
    torch.save(checkpoint, path)


# -----------------------------
# 11. Main Execution
# -----------------------------

def main():
    print(f"Using device: {DEVICE}")

    source_sentences, target_sentences = load_dataset(DATA_PATH)

    source_vocab = Vocabulary()
    target_vocab = Vocabulary()

    source_vocab.build_vocab(source_sentences, min_freq=MIN_FREQ)
    target_vocab.build_vocab(target_sentences, min_freq=MIN_FREQ)

    print(f"Number of sentence pairs: {len(source_sentences)}")
    print(f"Source vocab size: {len(source_vocab)}")
    print(f"Target vocab size: {len(target_vocab)}")

    encoder = Encoder(
        input_dim=len(source_vocab),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        model_type=MODEL_TYPE
    )

    decoder = Decoder(
        output_dim=len(target_vocab),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        model_type=MODEL_TYPE
    )

    model = Seq2Seq(
        encoder=encoder,
        decoder=decoder,
        target_vocab_size=len(target_vocab),
        device=DEVICE
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    criterion = nn.CrossEntropyLoss(
        ignore_index=target_vocab.word2idx[target_vocab.pad_token]
    )

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(
            model,
            source_sentences,
            target_sentences,
            source_vocab,
            target_vocab,
            optimizer,
            criterion
        )

        print(f"Epoch [{epoch}/{EPOCHS}], Loss: {train_loss:.4f}")

    save_checkpoint(model, source_vocab, target_vocab)

    print("\nSample translations:")
    test_sentences = [
        "how are you",
        "i am fine",
        "what is your name",
        "can you help me"
    ]

    for sentence in test_sentences:
        translation = translate_sentence(
            sentence,
            model,
            source_vocab,
            target_vocab
        )
        print(f"English: {sentence}")
        print(f"Hindi:   {translation}")
        print("-" * 40)


if __name__ == "__main__":
    main()