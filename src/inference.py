import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import os

from model import BurmeseDigitCNN_V2

def fit_and_pad(img: Image.Image, target_size: int = 28) -> Image.Image:
    """
    Sanitizes the image and mathematically pads it to maintain aspect ratio,
    preventing the CNN from squishing tall digits like ၇ and ၂.
    """
    img = img.convert("L")
    
    img = img.point(lambda p: 255 if p > 128 else 0)
    
    w, h = img.size
    if w > h:
        new_w = 20
        new_h = int(20 * (h / w))
    else:
        new_h = 20
        new_w = int(20 * (w / h))
        
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    delta_w = target_size - new_w
    delta_h = target_size - new_h
    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
    
    return ImageOps.expand(img, padding, fill=0)

def predict_digit(image_path: str, weights_path: str = "bhdr_v2_weights.pth") -> int:
    """
    Loads the V2 model and runs a forward pass on a single user image.
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Could not find weights at {weights_path}. Did you run train.py?")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = BurmeseDigitCNN_V2().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()
    
    raw_img = Image.open(image_path)
    processed_img = fit_and_pad(raw_img)
    
    transform = transforms.ToTensor()
    img_tensor = transform(processed_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs.data, 1)
        
    return predicted.item()

if __name__ == "__main__":
    test_image_path = "test.jpg"
    
    if os.path.exists(test_image_path):
        prediction = predict_digit(test_image_path)
        print(f"🤖 The V2 model predicts this digit is: {prediction}")
    else:
        print(f"Please place an image named '{test_image_path}' in the main folder to test.")