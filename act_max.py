import os
import numpy as np

import torch
from torch.optim import Adam
from torchvision import models

from misc_functions import preprocess_image, recreate_image, save_image
from simple_cnn import SimpleCNN


class CNNLayerVisualization():
    """
        Produces an image that minimizes the loss of a convolution
        operation for a specific layer and filter
    """
    def __init__(self, model, block, selected_layer, selected_filter, out_dir="generated"):
        self.model = model
        self.model.eval()
        self.selected_layer = selected_layer
        self.selected_filter = selected_filter
        self.conv_output = 0
        self.module = None
        for name, module in model.named_modules():
            if name==block:
                self.module = module
                break
        if self.module is None:
            raise Exception("block name doesn't exist in the model")
        self.out_dir = out_dir
        # Create the folder to export images if not exists
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    def hook_layer(self):
        def hook_function(module, grad_in, grad_out):
            # Gets the conv output of the selected filter (from selected layer)
            self.conv_output = grad_out[0, self.selected_filter]
        # Hook the selected layer
        assert isinstance(self.module, torch.nn.modules.container.Sequential), 'block is not sequential!!'
        self.module[self.selected_layer].register_forward_hook(hook_function)

    def visualise_layer_with_hooks(self):
        # Hook the selected layer
        self.hook_layer()
        # Generate a random image
        random_image = np.uint8(np.random.uniform(150, 180, (32, 32, 3)))
        # Process image and return variable
        processed_image = preprocess_image(random_image, False)
        # Define optimizer for the image
        optimizer = Adam([processed_image], lr=10, weight_decay=1e-6)
        for i in range(1, 257):
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            x = processed_image
            _ = self.model(x)
            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            assert isinstance(self.conv_output, torch.Tensor)
            loss = -torch.mean(self.conv_output)
            print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.numpy()))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image = recreate_image(processed_image)
            # Save image
            if i % 255 == 0:
                im_path = f'{self.out_dir}/layer_vis_l' + str(self.selected_layer) + \
                    '_f' + str(self.selected_filter) + '_iter' + str(i) + '.jpg'
                save_image(self.created_image, im_path)



if __name__ == '__main__':
    block = "block2"
    cnn_layer = 3
    for i in range(64):
        pretrained_model = SimpleCNN()
        pretrained_model.load_state_dict(torch.load('best_model.pth', map_location="cpu"))
        filter_pos = i
        # Fully connected layer is not needed
        layer_vis = CNNLayerVisualization(pretrained_model, block, cnn_layer, filter_pos, out_dir="block2_post_pool")

        # Layer visualization with pytorch hooks
        layer_vis.visualise_layer_with_hooks()
