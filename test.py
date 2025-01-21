import torch
import torch.nn.functional as F
import torch.utils.data.dataloader
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from PIL import ImageOps

from CNN_Class import CNN
import os

# Function to preprocess the image
def preprocess_image(image_path):
    try:
        # Open image and convert to grayscale
        image = Image.open(image_path).convert('L')
        
        # Ensure black digit on white background (invert if needed)
        # MNIST has white digits on black background
        img_array = np.array(image)
        mean_color = np.mean(img_array)
        if mean_color > 127:  # If background is white
            image = ImageOps.invert(image)
        
        transform = transforms.Compose([
            transforms.Resize((28, 28), antialias=True),
            transforms.ToTensor(),
            # MNIST images are white (1) on black (0)
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        image = transform(image)
        
        # Visualization before prediction
        plt.figure(figsize=(4,4))
        plt.imshow(image.squeeze(), cmap='gray')
        plt.title('Preprocessed Image')
        os.makedirs('./debug', exist_ok=True)
        plt.savefig('./debug/debug_preprocessed.png')
        plt.close()
        
        return image.unsqueeze(0)
    except Exception as e:
        print('Error loading or preprocessing image:', e)
        return None

def evaluate_on_mnist(model):
    # Load MNIST test dataset
    test_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=4)
    
    model.eval()
    correct = 0
    total = 0
    class_correct = [0] * 10
    class_total = [0] * 10
    
    with torch.no_grad():
        for data, targets in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            # Per-class accuracy
            for i in range(targets.size(0)):
                label = targets[i]
                pred = predicted[i]
                if label == pred:
                    class_correct[label] += 1
                class_total[label] += 1
    
    print(f'\nAccuracy on MNIST test set: {100 * correct / total:.2f}%')
    print('\nPer-class accuracy:')
    for i in range(10):
        print(f'Digit {i}: {100 * class_correct[i] / class_total[i]:.2f}%')
def main():
    # Load the trained model
    model = CNN()
    try:
        state_dict = torch.load('./models/cnn_model.pth', map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()
        # Add this before testing your custom image
        #print("Evaluating on MNIST test set...")
        #evaluate_on_mnist(model)
        
        # Debug model state
        #print("\nModel Architecture:")
        #print(model)
        print("\nFirst conv layer weights stats:")
        print(f"Mean: {model.conv1.weight.data.mean():.3f}")
        print(f"Std: {model.conv1.weight.data.std():.3f}")
    except Exception as e:
        print('Error loading model:', e)
        exit()

    # Load and preprocess the custom image
    image_path = './data/test/test.png'  # Replace with your image path
    image = preprocess_image(image_path)
    if image is None:
        exit()

    # Make predictions with debug info
    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)
        _, predicted = torch.max(output.data, 1)
        
        # Print detailed predictions
        print('\nDetailed predictions:')
        for digit, prob in enumerate(probabilities[0]):
            print(f'Digit {digit}: {prob:.4f}')
        
        print(f'\nRaw output before softmax: {output.squeeze().tolist()}')
        print(f'Predicted digit: {predicted.item()}')
        # Add visualization
        plt.imshow(image.squeeze(), cmap='gray')
        plt.title(f'Input Image (Predicted: {predicted.item()})')
        plt.show()

        # Add confidence check
        confidence = torch.max(probabilities).item()
        if confidence < 0.8:
            print(f"Warning: Low confidence prediction ({confidence:.2f})")

if __name__ == '__main__':
    main()
    