import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
from CNN_Class import CNN

# Function to preprocess the image
def preprocess_image(image_path):
    try:
        # Open image and convert to grayscale explicitly
        image = Image.open(image_path).convert('L')
        # Resize the image to match MNIST dimensions
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # Match MNIST normalization
        ])
        image = transform(image)
        # Debug prints
        print(f"Image shape: {image.shape}")
        print(f"Image min/max values: {image.min():.3f}/{image.max():.3f}")
        image = image.unsqueeze(0)
        return image
    except Exception as e:
        print('Error loading or preprocessing image:', e)
        return None

# Load the trained model
model = CNN()
try:
    state_dict = torch.load('./models/cnn_model.pth', map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    
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