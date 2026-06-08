import torch

from config import DEVICE
from data_loader import sample_episode

def test(model, test_data):
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()
    settings = [(5,1), (5,5), (10,1), (10,5)]
    with torch.no_grad():
        for way, shot in settings:
            c, t = 0, 0
            print(f"test {way}w{shot}s ...")
            for _ in range(1000):
                s, q, l, r = sample_episode(test_data, way, shot)
                logits = model(s, q, r)
                c += (logits.argmax(1).cpu() == torch.tensor(l)).sum().item()
                t += len(l)
            print(f"{way}w{shot}s: {c/t:.4f}")

if __name__ == "__main__":
