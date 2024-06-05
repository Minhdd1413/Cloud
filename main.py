import streamlit as st
from PIL import Image
import io
import torch
from torchvision import transforms
import torch.nn as nn

input_shape = (28, 28, 1)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = nn.functional.log_softmax(x, dim=1)
        return output
    
# Load your trained model
model = Net()
state_dict = torch.load('mnist_cnn.pth')
model.load_state_dict(state_dict)

st.write("""
# Simple Handwritten Digit Recognition App
Upload an image of a handwritten digit.
""")

file = st.file_uploader("Please upload an image file", type=["jpg", "png", "jpeg"])

def import_and_predict(image_data, model):
    size = (28,28)
    transform = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor(),
    transforms.Normalize(
        mean = [0.485],
        std = [0.229]
        )])
    
    image = Image.open(io.BytesIO(image_data))
    image = image.convert('L')  # Convert image to grayscale
    image = image.resize(size)
    
    batch_t = torch.unsqueeze(transform(image), 0)
    model.eval()
    out = model(batch_t)

    return torch.argmax(out).item()

if file is None:
    st.text("Please upload an image file")
else:
    file_bytes = file.read()
    prediction = import_and_predict(file_bytes, model)
    st.image(file_bytes, use_column_width=True)
    st.write("Predicted: ", prediction)