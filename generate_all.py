import os
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
from PIL import Image
import numpy as np
from simple_cnn import SimpleCNN
from find_top_filters import get_top_filters
import methods
import json
def main(model_path='best_model.pth', image_path='frog.jpg', output_dir='results', target_label=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    from torchvision import transforms
    from PIL import Image
    image = Image.open(image_path).convert('RGB')
    image_input = image.resize((32, 32))
    transform = transforms.Compose([transforms.ToTensor()])
    input_tensor = transform(image_input).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1).item()
    if target_label is not None and int(target_label) != prediction:
        print(f"VALIDATION_FAILED: Model predicted {prediction}, but ground truth is {target_label}")
        return
    print(f"Finding top filters for {image_path}...")
    top_filters = get_top_filters(model_path, image_path, percentile=90)
    block_order = {'block1': 0, 'block2': 1, 'block3': 2}
    top_filters.sort(key=lambda x: (block_order[x['block']], x['filter_index']))
    methods_list = [
        ('Method1_amu', methods.run_method1),
        ('Method2_actmax', methods.run_method2),
        ('Method3_lucent', methods.run_method3)
    ]
    all_images = {m[0]: {} for m in methods_list}
    for f in top_filters:
        block = f['block']
        idx = f['filter_index']
        layer_name = f"{block}.2" 
        print(f"Generating visualizations for {block} filter {idx}...")
        for name, run_fn in methods_list:
            try:
                img = run_fn(model, layer_name, idx)
                img_path = os.path.join(output_dir, f"{name}_{block}_{idx}.jpg")
                Image.fromarray(img).save(img_path)
                all_images[name][(block, idx)] = f"{name}_{block}_{idx}.jpg"
            except Exception as e:
                print(f"Error in {name} for {block} {idx}: {e}")
    graph_metadata = {}
    for name, _ in methods_list:
        print(f"Generating graph for {name}...")
        G = nx.DiGraph()
        nodes = []
        links = []
        nodes_by_block = {'block1': [], 'block2': [], 'block3': []}
        for f in top_filters:
            node_id = f"{f['block']}_{f['filter_index']}"
            G.add_node(node_id, block=f['block'], idx=f['filter_index'])
            nodes_by_block[f['block']].append(node_id)
            img_filename = all_images[name].get((f['block'], f['filter_index']))
            nodes.append({
                "id": node_id,
                "block": f['block'],
                "filter_index": f['filter_index'],
                "activation": f['activation'],
                "img": img_filename
            })
        for i in range(len(top_filters)):
            for j in range(i + 1, len(top_filters)):
                u = top_filters[i]
                v = top_filters[j]
                if block_order[v['block']] == block_order[u['block']] + 1:
                    u_id = f"{u['block']}_{u['filter_index']}"
                    v_id = f"{v['block']}_{v['filter_index']}"
                    G.add_edge(u_id, v_id)
                    links.append({"source": u_id, "target": v_id})
        graph_metadata[name] = {"nodes": nodes, "links": links}
        plt.figure(figsize=(15, 10))
        pos = {}
        for block, node_ids in nodes_by_block.items():
            x = block_order[block]
            for i, node_id in enumerate(node_ids):
                y = -(i - (len(node_ids) - 1) / 2)
                pos[node_id] = np.array([x, y])
        nx.draw(G, pos, with_labels=True, node_size=2500, node_color='white', edge_color='gray', font_size=7, arrowsize=15)
        ax = plt.gca()
        fig = plt.gcf()
        trans = ax.transData.transform
        trans2 = fig.transFigure.inverted().transform
        im_size = 0.04 
        for node in G.nodes():
            (block, idx) = (G.nodes[node]['block'], G.nodes[node]['idx'])
            img_filename = all_images[name].get((block, idx))
            if img_filename:
                img = np.array(Image.open(os.path.join(output_dir, img_filename)))
                (x, y) = pos[node]
                xx, yy = trans((x, y))
                xa, ya = trans2((xx, yy))
                a = plt.axes([xa-im_size/2, ya-im_size/2, im_size, im_size])
                a.imshow(img)
                a.set_xticks([])
                a.set_yticks([])
        plt.title(f"Filter Connectivity Graph (MLP Style) - {name}")
        graph_path = os.path.join(output_dir, f"graph_{name}.png")
        plt.savefig(graph_path)
        plt.close()
    with open(os.path.join(output_dir, 'graph_metadata.json'), 'w') as f:
        json.dump(graph_metadata, f, indent=4)
    print(f"Done! Results saved in {output_dir}")
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='best_model.pth')
    parser.add_argument('--image', type=str, default='frog.jpg')
    parser.add_argument('--output', type=str, default='results')
    parser.add_argument('--label', type=int, default=None)
    args = parser.parse_args()
    main(args.model, args.image, args.output, args.label)
