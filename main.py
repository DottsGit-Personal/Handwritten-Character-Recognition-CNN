import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
from CNN_Class import CNN

# Set device and random seed
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    cudnn.benchmark = True  # Enable cuDNN autotuner
    torch.cuda.empty_cache()  # Clear GPU cache
torch.manual_seed(42)

# Hyperparameters
num_epochs = 10  # Increased epochs
batch_size = 128
num_workers = 4
pin_memory = torch.cuda.is_available()
learning_rate = 0.001
weight_decay = 1e-4  # Increased weight decay

# Define transforms with augmentation
transform_train = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load and split datasets
full_train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform_train, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform_test, download=True)

# Split training into train and validation
train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

# Create data loaders
train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=True,  # Keep workers alive between epochs
    prefetch_factor=2  # Prefetch 2 batches per worker
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=True
)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

def warmup_dataloader(loader):
    """Warm up the dataloader by running one batch through it"""
    try:
        next(iter(loader))
    except StopIteration:
        pass

def cleanup_dataloaders(*loaders):
    """Properly cleanup DataLoader workers"""
    for loader in loaders:
        # Delete iterator reference
        if hasattr(loader, '_iterator'):
            del loader._iterator
        
        # Shutdown workers if they exist
        if hasattr(loader, '_worker_pids'):
            loader._shutdown_workers()

def main():
    try:
        print(f'Device: {device}')
        # Initialize model, optimizer, and scheduler
        model = CNN().to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
        criterion = nn.CrossEntropyLoss()

        # Warm up CUDA and DataLoaders
        print("Warming up...")
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
            # Run a small dummy tensor through the device
            dummy_tensor = torch.zeros(1, device=device)
            del dummy_tensor

        # Warm up dataloaders
        warmup_dataloader(train_loader)
        warmup_dataloader(val_loader)

        # Training loop
        best_val_loss = float('inf')
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            for batch_idx, (data, targets) in enumerate(train_loader):
                data = data.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
            # Validation phase
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for data, targets in val_loader:
                    data, targets = data.to(device), targets.to(device)
                    outputs = model(data)
                    val_loss += criterion(outputs, targets).item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            
            val_loss /= len(val_loader)
            accuracy = 100. * correct / total
            
            print(f'Epoch: {epoch+1}/{num_epochs}')
            print(f'Training Loss: {train_loss/len(train_loader):.4f}')
            print(f'Validation Loss: {val_loss:.4f}')
            print(f'Validation Accuracy: {accuracy:.2f}%')
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), './models/cnn_model.pth')

        print('Training completed!')
        print('Cleaning up...')

        # Explicit cleanup
        cleanup_dataloaders(train_loader, val_loader, test_loader)
            
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"Error during training: {e}")
        cleanup_dataloaders(train_loader, val_loader, test_loader)
    finally:
        # Force garbage collection
        import gc
        gc.collect()
    

if __name__ == '__main__':
    main()