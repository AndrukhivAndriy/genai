import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class DigitNN(nn.Module):
    def __init__(self, input_dim, num_hidden, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, num_hidden)
        self.layer2 = nn.Linear(num_hidden, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = nn.functional.relu(x)
        x = self.layer2(x)
        return x


# MNIST AutoDownload
def get_mnist_dataloaders(batch_size=32):
    transform = transforms.Compose([
        transforms.ToTensor(),  # Image to tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalization [-1, 1]
    ])

    # Loading train and test datasets
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    # Create DataLoader
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# Model config
model = DigitNN(28 * 28, 32, 10)
optimizer = optim.Adam(params=model.parameters(), lr=0.01)
loss_function = nn.CrossEntropyLoss()


train_data, test_data = get_mnist_dataloaders(batch_size=32)

# Learning
epochs = 2
model.train()

for epoch in range(epochs):
    loss_mean = 0
    lm_count = 0

    train_tqdm = tqdm(train_data, leave=True)
    for x_train, y_train in train_tqdm:
        x_train = x_train.view(x_train.size(0), -1)  # (batch, 28, 28) in (batch, 784)
        predict = model(x_train)
        loss = loss_function(predict, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lm_count += 1
        loss_mean = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * loss_mean
        train_tqdm.set_description(f"Epoch [{epoch+1}/{epochs}], loss_mean={loss_mean:.3f}")

# Tests
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for x_test, y_test in test_data:
        x_test = x_test.view(x_test.size(0), -1)  # (batch, 28, 28) у (batch, 784)
        outputs = model(x_test)
        predicted = torch.argmax(outputs, dim=1)
        correct += (predicted == y_test).sum().item()
        total += y_test.size(0)

accuracy = correct / total
print(f"Accuracy: {accuracy:.2%}")
