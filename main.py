from config import DEVICE
from data_loader import load_fewrel
from models import UltimateProtoNet
from train_and_test import train, test

train_data, val_data, test_data = load_fewrel()
model = UltimateProtoNet().to(DEVICE)
train(model, train_data)
test(model, test_data)

