import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

DATA_PATH = Path(__file__).resolve().parent / "data" / "pushkin.txt"
SAVE_DIR = Path(__file__).resolve().parent / "results"
SAVE_DIR.mkdir(exist_ok=True)

# параметры (не гигантская модель — на CPU тоже норм)
SEQ_LEN = 60
BATCH_SIZE = 64
HIDDEN = 256
N_LAYERS = 2
EPOCHS = 40
LR = 0.002
GEN_LEN = 350

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Устройство:", device)


def load_text():
    text = DATA_PATH.read_text(encoding="utf-8")
    # чуть подчистим лишние пробелы, но переносы строк оставим
    text = text.replace("\r\n", "\n")
    chars = sorted(list(set(text)))
    char2idx = {ch: i for i, ch in enumerate(chars)}
    idx2char = {i: ch for ch, i in char2idx.items()}
    print(f"Длина текста: {len(text)} символов, уникальных: {len(chars)}")
    return text, char2idx, idx2char


def make_batches(text, char2idx):
    data = [char2idx[c] for c in text]
    # нарезаем на куски длины SEQ_LEN+1 (вход + следующий символ)
    sequences = []
    # шаг поменьше — больше кусков на обучение (корпус короткий)
    step = 3
    for i in range(0, len(data) - SEQ_LEN - 1, step):
        sequences.append(data[i:i + SEQ_LEN + 1])

    random.shuffle(sequences)
    x = torch.tensor([s[:-1] for s in sequences], dtype=torch.long)
    y = torch.tensor([s[1:] for s in sequences], dtype=torch.long)
    return x, y


class CharLSTM(nn.Module):
    def __init__(self, vocab_size, hidden=HIDDEN, n_layers=N_LAYERS):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 128)
        self.lstm = nn.LSTM(128, hidden, n_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.lstm(x, hidden)
        logits = self.fc(out)
        return logits, hidden


def train_epoch(model, x_all, y_all, optimizer, criterion):
    model.train()
    # простой ручной батчинг
    perm = torch.randperm(x_all.size(0))
    total_loss = 0.0
    n = 0
    for i in range(0, x_all.size(0), BATCH_SIZE):
        idx = perm[i:i + BATCH_SIZE]
        xb = x_all[idx].to(device)
        yb = y_all[idx].to(device)

        optimizer.zero_grad()
        logits, _ = model(xb)
        loss = criterion(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
        loss.backward()
        # чуть клипаем градиенты, а то LSTM иногда взрывается
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        n += xb.size(0)
    return total_loss / n


@torch.no_grad()
def generate(model, char2idx, idx2char, seed="Мой дядя", length=GEN_LEN, temperature=0.8):
    model.eval()
    # если в seed есть неизвестный символ — пропускаем
    seed = "".join(ch for ch in seed if ch in char2idx)
    if not seed:
        seed = "А"

    ids = [char2idx[c] for c in seed]
    result = list(seed)
    hidden = None
    inp = torch.tensor([ids], dtype=torch.long, device=device)

    # прогоняем seed целиком
    logits, hidden = model(inp, hidden)
    next_id = ids[-1]

    for _ in range(length):
        inp = torch.tensor([[next_id]], dtype=torch.long, device=device)
        logits, hidden = model(inp, hidden)
        probs = torch.softmax(logits[0, -1] / temperature, dim=-1)
        next_id = torch.multinomial(probs, 1).item()
        result.append(idx2char[next_id])

    return "".join(result)


def main():
    text, char2idx, idx2char = load_text()
    x_all, y_all = make_batches(text, char2idx)
    print("Число последовательностей:", x_all.size(0))

    model = CharLSTM(len(char2idx)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    losses = []
    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, x_all, y_all, optimizer, criterion)
        losses.append(loss)
        if epoch % 5 == 0 or epoch == 1:
            sample = generate(model, char2idx, idx2char, seed="Онегин", length=180, temperature=0.85)
            print(f"\n=== Epoch {epoch}, loss={loss:.4f} ===")
            print(sample[:300])
            print("...")

    # сохранение
    torch.save({
        "model_state": model.state_dict(),
        "char2idx": char2idx,
        "idx2char": {int(k): v for k, v in idx2char.items()},
        "hidden": HIDDEN,
        "n_layers": N_LAYERS,
    }, SAVE_DIR / "char_lstm.pth")

    with open(SAVE_DIR / "history.json", "w", encoding="utf-8") as f:
        json.dump({"loss": losses}, f, indent=2)

    plt.figure(figsize=(7, 4))
    plt.plot(losses)
    plt.title("Loss при обучении LSTM")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "loss.png", dpi=120)
    plt.close()

    # финальные примеры генерации
    seeds = ["Мой дядя", "Зима", "У лукоморья", "Я вас любил"]
    examples = []
    for s in seeds:
        text_out = generate(model, char2idx, idx2char, seed=s, length=GEN_LEN, temperature=0.75)
        examples.append({"seed": s, "text": text_out})
        print("\n--- seed:", s, "---")
        print(text_out)

    with open(SAVE_DIR / "generated_examples.txt", "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(f"SEED: {ex['seed']}\n")
            f.write(ex["text"])
            f.write("\n\n" + "=" * 50 + "\n\n")

    print("\nГотово. Результаты в", SAVE_DIR)


if __name__ == "__main__":
    main()
