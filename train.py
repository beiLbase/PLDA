import torch
import torch.nn as nn
import torch.optim as optim

from config import DEVICE
from data_loader import sample_episode

def train(model, train_data):
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0

    for epoch in range(15):
        model.train()
        correct, total, total_loss = 0, 0, 0.0

        print(f"Epoch {epoch+1}/15 ")
        for _ in range(250):
            support, query, labels, rel_descs = sample_episode(train_data)
            logits = model(support, query, rel_descs)
            loss = criterion(logits, torch.tensor(labels).to(DEVICE))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            correct += (logits.argmax(1).cpu() == torch.tensor(labels)).sum().item()
            total += len(labels)
            total_loss += loss.item()
        acc = correct / total
        print(f"Acc: {acc:.4f} Loss: {total_loss/250:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "best_model.pth")

if __name__ == "__main__":

    pass