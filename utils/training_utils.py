import torch

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    train_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    return train_loss / len(loader)



def evaluate_model(model, loader, criterion, device):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for X_test, y_test in loader:
            X_test, y_test = X_test.to(device), y_test.to(device)
            outputs = model(X_test)
            test_loss += criterion(outputs, y_test).item()

    return test_loss / len(loader)