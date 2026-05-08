import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from simple_cnn import SimpleCNN
def get_top_filters(model_path, image_path, percentile=90):
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    image = Image.open(image_path).convert('RGB')
    image = image.resize((32, 32))
    transform = transforms.Compose([transforms.ToTensor()])
    input_tensor = transform(image).unsqueeze(0)
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
    model.block1[2].register_forward_hook(get_activation('block1'))
    model.block2[2].register_forward_hook(get_activation('block2'))
    model.block3[2].register_forward_hook(get_activation('block3'))
    with torch.no_grad():
        model(input_tensor)
    filter_activations = []
    for block_name in ['block1', 'block2', 'block3']:
        act = activations[block_name]  
        mean_act = torch.mean(act, dim=(2, 3)).squeeze() 
        for i, val in enumerate(mean_act):
            filter_activations.append({
                'block': block_name,
                'filter_index': i,
                'activation': val.item()
            })
    filter_activations.sort(key=lambda x: x['activation'], reverse=True)
    num_filters = len(filter_activations)
    top_n = int(num_filters * (100 - percentile) / 100)
    top_filters = filter_activations[:top_n]
    return top_filters
if __name__ == "__main__":
    top_filters = get_top_filters('best_model.pth', 'frog.jpg')
    for f in top_filters:
        print(f"Block: {f['block']}, Filter: {f['filter_index']}, Activation: {f['activation']:.4f}")
