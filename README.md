# Activation Maximization Experiments

This project explores three different implementations of Activation Maximization to visualize the features learned by a Convolutional Neural Network (CNN). Specifically, it targets a `SimpleCNN` model trained on CIFAR-10.

## Overview

The goal is to identify which convolutional filters are most active for a given input image (e.g., a frog) and then generate synthetic images that maximize the activation of those specific filters using different techniques.

### Included Methods

1.  **Method 1 (act_max_util)**: A custom implementation using various regularizers such as L2 decay, Gaussian blur, and pixel clipping to produce cleaner visualizations.
2.  **Method 2 (act_max)**: A standard gradient ascent approach using the Adam optimizer to maximize the mean activation of a target filter.
3.  **Method 3 (Lucent)**: Utilizes the `lucent` library, a high-quality visualization tool inspired by OpenAI's Clarity.

## Project Structure

- `generate_all.py`: The main orchestration script.
- `find_top_filters.py`: Logic to find the top 10% most active filters for an input image.
- `methods.py`: Unified API wrapper for the three activation maximization techniques.
- `simple_cnn.py`: Definition of the CNN architecture.
- `best_model.pth`: Pre-trained weights for the SimpleCNN.

## Installation

Ensure you have a Python environment set up, then install the dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Generate Data
To run the full pipeline (find top filters, generate visualizations, and create connectivity graphs):

```bash
python3 generate_all.py --image frog.jpg --model best_model.pth --output results
```

### 2. Launch Interactive UI
The project includes a React + FastAPI interface for an interactive exploration of the filter graph.

**Features:**
- **Image Upload**: Upload any image (e.g., JPEG, PNG) to dynamically identify its top activating filters.
- **On-the-fly Generation**: Trigger the full activation maximization pipeline directly from the browser.
- **Interactive Graph**: Pan, zoom, and drag nodes. Each node displays the synthesized filter image.
- **Method Comparison**: Toggle between AMU, ActMax, and Lucent visualizations for the same filters.

#### Start the Backend:
```bash
python3 ui/server/main.py
```

#### Start the Frontend:
```bash
cd ui/client
npm install
npm run dev
```
Open [http://localhost:5173](http://localhost:5173). Use the **"Upload Image"** button to start a new analysis. A loading overlay will appear while the backend processes the image (which can take 1-3 minutes).

## Results
...

After running the script, the specified output directory will contain:
- Individual `.jpg` visualizations for each selected filter and each method.
- `graph_Method1_amu.png`: Connectivity graph using Method 1 images.
- `graph_Method2_actmax.png`: Connectivity graph using Method 2 images.
- `graph_Method3_lucent.png`: Connectivity graph using Lucent images.

The nodes in the graphs are the selected filters, and edges represent their sequential connections within the network.

## Credits
Some of this code in this repository has been adapted and modified from these three repositories:
https://github.com/greentfrapp/lucent
https://github.com/utkuozbulak/pytorch-cnn-visualizations
https://github.com/Nguyen-Hoa/Activation-Maximization
