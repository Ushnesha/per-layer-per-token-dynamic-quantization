import torch
import torch.nn as nn

class ShallowMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        return self.layer3(self.relu(self.layer2(self.relu(self.layer1(x)))))

def train(model, inputs, targets, optim, loss_fn, epochs):
    model.train()
    for epoch in range(epochs):
        optim.zero_grad()
        preds = model(inputs).squeeze(-1)
        loss = loss_fn(preds, targets)
        loss.backward()
        optim.step()

        if epoch % 200 == 0:
            print(f"Epoch: {epoch} | Loss: {loss.item():.4f}")

