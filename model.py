import torch
import torch.nn as nn
import torch.optim as optim
import json
from datetime import datetime
from sklearn.model_selection import train_test_split

# check if CUDA is available and set device accordingly
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("CUDA is available")
else:
    device = torch.device('cpu')
    print("CUDA is not available. Switching to CPU.")

# Install PyTorch with CUDA support if necessary
if device.type == 'cuda':
    try:
        # Check if torch with CUDA support is already installed
        torch.zeros(1).cuda()
    except:
        # If not, install PyTorch with CUDA support
        import subprocess

        subprocess.call(
            "pip install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu111/torch_stable.html",
            shell=True)

# Load data from json file
with open('small_data.json', 'r') as f:
    data = json.load(f)

# Convert data to PyTorch tensors
inputs = torch.tensor([data[d]['embedded'] for d in data])
labels = torch.tensor([0 if data[d]['genre'] == 'Drama' else 1 for d in data])
# Partition data into training and testing sets
inputs_train, inputs_test, labels_train, labels_test = train_test_split(inputs, labels, test_size=0.2)

# Define the neural network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x


# Create an instance of the model
net = Net()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Train the model
for epoch in range(10):
    running_loss = 0.0
    for i in range(len(inputs_train)):
        optimizer.zero_grad()
        outputs = net(inputs_train[i])
        loss = criterion(outputs.unsqueeze(0), labels_train[i].unsqueeze(0))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    now = datetime.now()
    print(now.strftime("%d/%m/%Y %H:%M:%S")+' Epoch %d loss: %.3f' % (epoch + 1, running_loss / len(inputs_train)))

# Test the model on the training set
correct = 0
total = 0
with torch.no_grad():
    for i in range(len(inputs_train)):
        outputs = net(inputs_train[i])
        _, predicted = torch.max(outputs.data, 0)
        total += 1
        correct += (predicted == labels_train[i]).sum().item()

print('Accuracy on training set: %.3f%%' % (100 * correct / total))

# Test the model on the testing set
correct = 0
total = 0
with torch.no_grad():
    for i in range(len(inputs_test)):
        outputs = net(inputs_test[i])
        _, predicted = torch.max(outputs.data, 0)
        total += 1
        correct += (predicted == labels_test[i]).sum().item()

print('Accuracy on testing set: %.3f%%' % (100 * correct / total))