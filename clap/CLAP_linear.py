import torch

class classifier(torch.nn.Module):
    def __init__(self, hidden_dim, num_labels):
        # clap hidden = 1024, labels = 2, 3 input per sample, hidden_dim=1024*3=3072
        super(classifier, self).__init__()
        self.layer1 = torch.nn.Linear(hidden_dim, num_labels)

    def forward(self, x):
        # Accept tensors shaped (batch, seq_len, hidden_dim) by pooling across the sequence axis

        # Align tensor dtype with the classifier weights to avoid dtype mismatches
        x = x.to(self.layer1.weight.dtype)

        x = self.layer1(x)
        return x

def train_model(model, dataloader, optimizer, criterion, device, epochs=5):
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for batch_inputs, batch_labels in dataloader:
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            logits = model(batch_inputs)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"epoch {epoch}: loss={avg_loss:.4f}")