import torch
from torch.utils.data import DataLoader
import CLAP_linear
import data_load_individual 
import data_load
import re

trainFOLDER_PATTERN = re.compile(r"t_(\d+)$", re.IGNORECASE)
testFOLDER_PATTERN = re.compile(r"v_(\d+)$", re.IGNORECASE)

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # get train dataset
    train_root = "\datasets\EATD-Corpus\CLAP"
    train_label_root = "\datasets\EATD-Corpus\EATD-Corpus"
    data_load.copy_label_files(train_label_root, train_root, trainFOLDER_PATTERN)
    train_dataset = data_load_individual._load_split(train_root, trainFOLDER_PATTERN)
    print(f"Train dataset size: {len(train_dataset)} samples")

    # get test dataset
    test_root = "\datasets\EATD-Corpus\CLAP"
    test_label_root = "\datasets\EATD-Corpus\EATD-Corpus"
    data_load.copy_label_files(test_label_root, test_root, testFOLDER_PATTERN)
    test_dataset = data_load_individual._load_split(test_root, testFOLDER_PATTERN)
    print(f"Test dataset size: {len(test_dataset)} samples")
    
    # get dataloader
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # define model
    model = CLAP_linear.classifier(1024, 2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    # train model 
    for epoch in range(100):
        model.train()
        running_loss = 0.0
        
        for batch_inputs, batch_labels in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            logits = model(batch_inputs)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"epoch {epoch+1}: loss={running_loss/len(train_loader):.4f}")
        
    # evaluate model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        
        for batch_inputs, batch_labels in test_loader:
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)
            
            logits = model(batch_inputs)
            _, predicted = torch.max(logits, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
        
        print(f"Test Accuracy: {100 * correct / total:.2f}%")