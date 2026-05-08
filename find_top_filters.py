import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
from simple_cnn import SimpleCNN

def get_model(model_type, model_path=None):
    if model_type == 'simple_cnn':
        model = SimpleCNN()
        if model_path:
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
        layers = ['block1.2', 'block2.2', 'block3.2']
    elif model_type == 'alexnet':
        model = models.alexnet(pretrained=True)
        layers = ['features.1', 'features.4', 'features.7', 'features.9', 'features.11']
    elif model_type == 'vgg16':
        model = models.vgg16(pretrained=True)
        layers = ['features.1', 'features.3', 'features.6', 'features.8', 'features.11', 'features.13', 'features.15']
    elif model_type == 'inception_v1':
        from lucent.modelzoo import inceptionv1
        model = inceptionv1(pretrained=True)
        layers = ['conv2d0_pre_relu_conv', 'mixed3a', 'mixed3b', 'mixed4a', 'mixed4b', 'mixed4c']
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return model, layers

def get_top_filters(model_type, image_path, model_path=None, percentile=90, input_size=32):
    model, layers = get_model(model_type, model_path)
    model.eval()
    image = Image.open(image_path).convert('RGB')
    
    if model_type == 'simple_cnn':
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    input_tensor = transform(image).unsqueeze(0)
    activations = {}
    def get_activation(name):
        def hook(m, i, o):
            activations[name] = o.detach()
        return hook
    handles = []
    for name in layers:
        found = False
        for n, m in model.named_modules():
            if n == name:
                handles.append(m.register_forward_hook(get_activation(name)))
                found = True
                break
        if not found:
            print(f"Warning: Layer {name} not found in model")
    with torch.no_grad():
        model(input_tensor)
    for h in handles:
        h.remove()
    filter_activations = []
    for name in layers:
        if name not in activations: continue
        act = activations[name]
        mean_act = torch.mean(act, dim=(2, 3)).squeeze()
        if mean_act.dim() == 0: mean_act = mean_act.unsqueeze(0)
        for i, val in enumerate(mean_act):
            filter_activations.append({
                'block': name,
                'filter_index': i,
                'activation': val.item()
            })
    filter_activations.sort(key=lambda x: x['activation'], reverse=True)
    num_filters = len(filter_activations)
    top_n = max(1, int(num_filters * (100 - percentile) / 100))
    return filter_activations[:top_n]

if __name__ == "__main__":
    top_filters = get_top_filters('simple_cnn', 'frog.jpg', 'best_model.pth', input_size=32)
    for f in top_filters:
        print(f"Layer: {f['block']}, Filter: {f['filter_index']}, Activation: {f['activation']:.4f}")
