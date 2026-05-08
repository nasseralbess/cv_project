import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import os
from PIL import Image
from torchvision import transforms
from lucent.optvis import render, param, transform, objectives
import act_max_util as amu
from misc_functions import preprocess_image, recreate_image, save_image

class Method2Visualization:
    def __init__(self, model, layer_name, selected_filter, input_size=32):
        self.model = model
        self.model.eval()
        self.layer_name = layer_name
        self.selected_filter = selected_filter
        self.conv_output = 0
        self.module = None
        self.input_size = input_size
        for name, module in model.named_modules():
            if name == layer_name:
                self.module = module
                break
        if self.module is None:
            raise Exception(f"Layer {layer_name} not found")

    def hook_layer(self):
        def hook_function(module, grad_in, grad_out):
            self.conv_output = grad_out[0, self.selected_filter]
        return self.module.register_forward_hook(hook_function)

    def run(self, steps=256):
        handle = self.hook_layer()
        random_image = np.uint8(np.random.uniform(150, 180, (self.input_size, self.input_size, 3)))
        processed_image = preprocess_image(random_image, False)
        optimizer = Adam([processed_image], lr=0.1, weight_decay=1e-6)
        for i in range(1, steps + 1):
            optimizer.zero_grad()
            self.model(processed_image)
            loss = -torch.mean(self.conv_output)
            loss.backward()
            optimizer.step()
        handle.remove()
        return recreate_image(processed_image)

def run_method1(model, layer_name, filter_index, steps=200, input_size=32):
    activation_dictionary = {}
    module = dict(model.named_modules())[layer_name]
    handle = module.register_forward_hook(amu.layer_hook(activation_dictionary, layer_name))
    input_img = torch.randn(1, 3, input_size, input_size, requires_grad=True)
    optimized_img = amu.act_max(
        network=model,
        input=input_img,
        layer_activation=activation_dictionary,
        layer_name=layer_name,
        unit=filter_index,
        steps=steps,
        alpha=torch.tensor(100),
        conv=True,
        L2_Decay=True,
        Gaussian_Blur=True,
        Norm_Crop=True,
        Contrib_Crop=True
    )
    handle.remove()
    img = amu.image_converter(optimized_img.detach().squeeze(0))
    return (img * 255).astype(np.uint8)

def run_method2(model, layer_name, filter_index, steps=256, input_size=32):
    vis = Method2Visualization(model, layer_name, filter_index, input_size=input_size)
    return vis.run(steps)

def run_method3(model, layer_name, filter_index, steps=256, input_size=32):
    lucent_layer = f"{layer_name.replace('.', '_')}:{filter_index}"
    param_f = lambda: param.image(input_size)
    imgs = render.render_vis(model, lucent_layer, param_f=param_f, transforms=[], thresholds=(steps,), show_image=False, progress=False)
    img = imgs[0]
    if len(img.shape) == 4:
        img = img[0]
    return (img * 255).astype(np.uint8)
