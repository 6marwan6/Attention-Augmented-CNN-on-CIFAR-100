import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from aaconv import AugmentedConv 


# 1. Load CIFAR-100

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))  
])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)


# 2. Define Augmented CNN

class SimpleAugmentedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # First convolutional layer: standard 2D convolution with 64 output channels
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        # Batch normalization layer 
        self.bn1 = nn.BatchNorm2d(64)
        
        # Attention-augmented convolutional block
        self.augmented_block = AugmentedConv(
            in_channels=64,  # Input channels from the previous layer
            out_channels=128,  # Output channels for this layer
            kernel_size=3,  # Size of the convolutional kernel
            dk=40,  # Dimension of the key
            dv=40,  # Dimension of the value
            Nh=4,  # Number of attention heads
            relative=True  # Use relative positional encoding
        )

        #  reduce the spatial dimensions to 1x1
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # fully connected layer to map the features to 100 classes 
        self.fc = nn.Linear(128, 100)

    def forward(self, x):
        # Apply the first convolutional layer followed by ReLU activation and batch normalization
        x = torch.relu(self.bn1(self.conv1(x)))
       
        x = self.augmented_block(x)
        
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)
       
        return self.fc(x)


# 3. train the Model

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleAugmentedCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Training...")
    for epoch in range(10):  # 10 epochs
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print(f"[Epoch {epoch+1}, Batch {i+1}] loss: {running_loss / 100:.3f}")
                running_loss = 0.0

# 3. evaluation
    print("Evaluating...")
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy on CIFAR-100: {100 * correct / total:.2f}%")
