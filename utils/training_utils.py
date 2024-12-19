import torch

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    train_loss = 0.0
    for _, (X, y) in enumerate(loader):
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    return train_loss / len(loader)



def evalute_model(model, loader, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_val, y_val in loader:
            X_val, y_val = X_val.to(device), y_val.to(device)
            val_outputs = model(X_val)
            val_loss += criterion(val_outputs, y_val).item()

    return val_loss / len(loader)