import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import os

# --- 1. Define a Simple Convolutional Neural Network (CNN) ---
class AnimalClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(AnimalClassifier, self).__init__()
        # Input: 3 channels (RGB), 128x128 pixels
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # -> 32 x 64 x 64
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # -> 64 x 32 x 32
        )
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # -> 128 x 16 x 16
        )
        
        # Flatten the output for the linear layers
        # 128 channels * 16 * 16 = 32768
        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = self.fc_layer(x)
        return x

# --- 2. Setup Data, Training, and Evaluation ---
def main():
    # --- Hyperparameters ---
    learning_rate = 0.001
    batch_size = 32
    num_epochs = 10
    data_dir = './data' # This is the folder Git LFS will track
    
    # --- Data Preprocessing ---
    # Define transformations for the images
    # We resize all images to be the same size for the network
    data_transforms = transforms.Compose([
        transforms.Resize((128, 128)), # Resize to 128x128
        transforms.ToTensor(),         # Convert to PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], # Normalize (standard for ImageNet)
                             std=[0.229, 0.224, 0.225])
    ])

    # --- Load Dataset ---
    # ImageFolder expects data to be in:
    # data/
    #   ├── flying/
    #   │   ├── img1.jpg
    #   │   ├── img2.jpg
    #   │   └── ...
    #   └── not_flying/
    #       ├── imgA.jpg
    #       ├── imgB.jpg
    #       └── ...
    
    try:
        full_dataset = ImageFolder(data_dir, transform=data_transforms)
        print(f"Dataset loaded. Found {len(full_dataset)} images.")
        print(f"Classes: {full_dataset.classes}") # Should show ['flying', 'not_flying']
    except FileNotFoundError:
        print(f"Error: Data directory '{data_dir}' not found.")
        print("Please create it and add your 'flying' and 'not_flying' subfolders.")
        return
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # --- Split into Training and Validation ---
    val_split = 0.2 # Use 20% for validation
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # --- Initialize Model, Loss, and Optimizer ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = AnimalClassifier(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # --- Training Loop ---
    for epoch in range(num_epochs):
        model.train() # Set model to training mode
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {running_loss/len(train_loader):.4f}')
        
        # --- Validation Loop ---
        model.eval() # Set model to evaluation mode
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad(): # No need to calculate gradients for validation
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss/len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.2f} %')

    print('Finished Training')
    
    # --- Save the Model ---
    # (Optional)
    # torch.save(model.state_dict(), 'animal_classifier_model.pth')
    # print('Model saved to animal_classifier_model.pth')

if __name__ == '__main__':
    main()
