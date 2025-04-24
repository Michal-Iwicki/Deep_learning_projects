import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_Ensemble(nn.Module):
    def __init__(self, n, w0=1, w1=64, w2=128, w3=256, dr=0.25, use_bn=True):
        super(CNN_Ensemble, self).__init__()

        self.conv1d_1 = nn.Conv1d(w0, w1, kernel_size=3, padding=1)
        self.conv1d_2 = nn.Conv1d(w1, w2, kernel_size=3, padding=1)
        self.conv1d_3 = nn.Conv1d(w2, w3, kernel_size=3, padding=1)
        self.bn1d_1 = nn.BatchNorm1d(w1) if use_bn else nn.Identity()
        self.bn1d_2 = nn.BatchNorm1d(w2) if use_bn else nn.Identity()
        self.bn1d_3 = nn.BatchNorm1d(w3) if use_bn else nn.Identity()
        self.pool1d = nn.MaxPool1d(2)

        self.conv2d_1 = nn.Conv2d(w0, w1, kernel_size=3, padding=1)
        self.conv2d_2 = nn.Conv2d(w1, w2, kernel_size=3, padding=1)
        self.conv2d_3 = nn.Conv2d(w2, w3, kernel_size=3, padding=1)
        self.bn2d_1 = nn.BatchNorm2d(w1) if use_bn else nn.Identity()
        self.bn2d_2 = nn.BatchNorm2d(w2) if use_bn else nn.Identity()
        self.bn2d_3 = nn.BatchNorm2d(w3) if use_bn else nn.Identity()
        self.pool2d = nn.MaxPool2d((2, 2))

        self.drop = nn.Dropout(p=dr)
        self.act = F.relu
        self.out_act_1 = nn.Linear(532480, 128)
        self.out_act_2 = nn.Linear(128, n)

        return

    def forward(self, x):
        raw, mel = x

        raw = self.pool1d(self.act(self.bn1d_1(self.conv1d_1(raw))))
        raw = self.pool1d(self.act(self.bn1d_2(self.conv1d_2(raw))))
        raw = self.pool1d(self.act(self.bn1d_3(self.conv1d_3(raw))))
        raw = torch.flatten(raw, 1)

        mel = self.pool2d(self.act(self.bn2d_1(self.conv2d_1(mel))))
        mel = self.pool2d(self.act(self.bn2d_2(self.conv2d_2(mel))))
        mel = self.pool2d(self.act(self.bn2d_3(self.conv2d_3(mel))))
        mel = torch.flatten(mel, 1)

        concatenated = torch.cat((raw, mel), 1)
        concatenated = self.act(self.out_act_1(self.drop(concatenated)))
        concatenated = self.out_act_2(self.drop(concatenated))

        return concatenated


def train_model(model, train_loader, validation_loader, optimizer, epochs=10, patience=3, tracking=False, printer=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    best_validation_loss = float("+inf")
    criterion = nn.CrossEntropyLoss()
    
    counter = 0
    train_losses = []
    validation_losses = []
    
    for epoch in range(epochs):
        if counter >= patience:
            print("Patience triggered. End of learning")

            break
            
        model.train()
        running_loss = 0.0
        for (raw, mel), labels in train_loader:
            raw, mel, labels = raw.to(device), mel.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model((raw, mel))
            loss = criterion(outputs, labels)
            loss.backward()
            
            optimizer.step()
            running_loss += loss.item()
        
        model.eval()
        validation_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for (raw, mel), labels in validation_loader:
                raw, mel, labels = raw.to(device), mel.to(device), labels.to(device)
                
                outputs = model((raw, mel))
                loss = criterion(outputs, labels)
                
                validation_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        validation_loss /= len(validation_loader)
        accuracy = 100 * correct / total

        if tracking:
            train_losses.append(running_loss / len(train_loader))
            validation_losses.append(validation_loss)
            
        if printer:
            print(f"Epoch {epoch + 1}/{epochs}, "
                  f"Loss: {running_loss / len(train_loader):.4f}, "
                  f"Validation Loss: {validation_loss:.4f}, "
                  f"Validation Accuracy: {accuracy:.2f}%")

        if validation_loss < best_validation_loss:
            best_validation_loss_loss = validation_loss
            best_model = model.state_dict()

            counter=0
        else:
            counter += 1

    model.load_state_dict(best_model)
    
    if tracking:
        return train_losses, validation_losses

def evaluate(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    criterion = nn.CrossEntropyLoss()

    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for (raw, mel), labels in test_loader:
            raw, mel, labels = raw.to(device), mel.do(device), labels.to(device)

            outputs = model((raw, mel))
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = 100 * correct / total

    return test_loss, accuracy