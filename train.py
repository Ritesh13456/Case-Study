import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from model import PrunableNet
from utils import compute_sparsity_loss, compute_sparsity
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Dataset
transform = transforms.Compose([
    transforms.ToTensor()
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform
)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform
)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64)

# Model
model = PrunableNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

lambda_ = 0.1

# Training
for epoch in range(5):
    model.train()
    total_loss = 0

    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        cls_loss = F.cross_entropy(outputs, labels)
        sp_loss = compute_sparsity_loss(model)

        loss = cls_loss + lambda_ * sp_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.2f}")

# Evaluation
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
sparsity = compute_sparsity(model)

print(f"\nFinal Accuracy: {accuracy:.2f}%")
print(f"Sparsity Level: {sparsity:.2f}%")

#Plot gate distribution
gates_all = []

for m in model.modules():
    if hasattr(m, 'gate_scores'):
        gates = torch.sigmoid(m.gate_scores / 0.5).detach().cpu().numpy()
        gates_all.extend(gates.flatten())

plt.hist(gates_all, bins=50)
plt.title("Gate Distribution")
plt.xlabel("Gate Value")
plt.ylabel("Frequency")
plt.show()