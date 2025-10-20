import torch 
import os
import itertools
import re
import shutil

# data preprocess and loader
class dep_data(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    
def load_feature(path):
    # pool embedding to shape(hidden_dim,)
    x = torch.load(path)

    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Expected tensor at {path}, got type {type(x)}")

    if x.dim() <= 1:
        return x.view(-1)

    hidden_size = x.size(-1)
    x = x.reshape(-1, hidden_size)
    return x.mean(dim=0)

def _iter_tensor_folders(root_dir, pattern):
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Root directory not found: {root_dir}")

    matching = [name for name in os.listdir(root_dir) if pattern.fullmatch(name)]
    for name in sorted(matching, key=lambda n: int(pattern.fullmatch(n).group(1))):
        folder_path = os.path.join(root_dir, name)
        if os.path.isdir(folder_path):
            yield name, folder_path

def _load_split(root_dir, pattern):
    features, labels = [], []

    for _, folder in _iter_tensor_folders(root_dir, pattern):
        feats = {
            emotion: load_feature(os.path.join(folder, f"{emotion}.pt"))
            for emotion in ["pos_hidden_state", "neg_hidden_state", "nue_hidden_state"]
        }

        with open(os.path.join(folder, "new_label.txt")) as f:
            score = float(f.read().strip())

        depressed = score >= 53
        label = 1 if depressed else 0

        samples = get_rearranged_samples(feats, depressed)
        for sample in samples:
            features.append(sample)
            labels.append(label)

    if not features:
        raise ValueError(f"No tensor samples found under {root_dir}")

    stacked_features = torch.stack(features)
    tensor_labels = torch.tensor(labels, dtype=torch.long)
    return dep_data(stacked_features, tensor_labels)

def _iter_matching_folders(root_dir: str, pattern: re.Pattern):
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Root directory not found: {root_dir}")

    for name in sorted(os.listdir(root_dir)):
        if pattern.fullmatch(name):
            folder_path = os.path.join(root_dir, name)
            if os.path.isdir(folder_path):
                yield name, folder_path

def copy_label_files(label_root: str, target_root: str, pattern: re.Pattern):
    for name, target_path in _iter_matching_folders(target_root, pattern):
        label_folder = os.path.join(label_root, name)
        label_file = os.path.join(label_folder, "new_label.txt")

        if not os.path.exists(label_folder):
            print(f"Warning: label folder missing for {name}; skipping")
            continue

        if not os.path.isfile(label_file):
            print(f"Warning: label file missing in {label_folder}; skipping")
            continue

        dest_file = os.path.join(target_path, "new_label.txt")
        os.makedirs(target_path, exist_ok=True)
        shutil.copy2(label_file, dest_file)
        
# pre-process and resample
def get_rearranged_samples(features, depressed=True):
    # rearrange depressed samples
    if depressed:
        orders = list(itertools.permutations(['pos_hidden_state', 'neg_hidden_state', 'nue_hidden_state']))
    else:
        orders = [['pos_hidden_state', 'neg_hidden_state', 'nue_hidden_state']]  # only one order
    samples = []
    for order in orders:
        combined = torch.cat([features[o] for o in order])  # concat in order
        samples.append(combined)
    return samples

