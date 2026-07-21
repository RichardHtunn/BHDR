import torch
from PIL import Image
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import BurmeseDigitCNN_V2
from src.inference import fit_and_pad

def test_fit_and_pad_size() -> None:
    """
    Tests if the padding function strictly returns a 28x28 image, 
    even if the user uploads a weirdly shaped image.
    """
    dummy_img = Image.new('RGB', (100, 50), color='white')
    
    processed_img = fit_and_pad(dummy_img, target_size=28)
    
    assert processed_img.size == (28, 28), f"Expected (28, 28), got {processed_img.size}"

def test_model_output_shape() -> None:
    """
    Tests if the V2 CNN architecture takes a 1x28x28 image 
    and outputs exactly 10 classes (digits 0-9).
    """
    model = BurmeseDigitCNN_V2()
    
    dummy_tensor = torch.randn(1, 1, 28, 28)
    
    output = model(dummy_tensor)
    
    assert output.shape == (1, 10), f"Expected shape [1, 10], got {output.shape}"