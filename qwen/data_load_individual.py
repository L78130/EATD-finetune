import os
import re
import torch

EMOTION_FILES = ("pos_hidden_state", "neg_hidden_state", "nue_hidden_state")


class dep_data(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def load_feature(path):
    tensor = torch.load(path)

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected tensor at {path}, got type {type(tensor)}")

    if tensor.dim() <= 1:
        return tensor.view(-1)

    hidden_size = tensor.size(-1)
    tensor = tensor.reshape(-1, hidden_size)
    return tensor.mean(dim=0)


def _iter_tensor_folders(root_dir: str, pattern: re.Pattern):
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Root directory not found: {root_dir}")

    matching = [name for name in os.listdir(root_dir) if pattern.fullmatch(name)]
    for name in sorted(matching, key=lambda n: int(pattern.fullmatch(n).group(1))):
        folder_path = os.path.join(root_dir, name)
        if os.path.isdir(folder_path):
            yield name, folder_path


def _load_split(root_dir: str, pattern: re.Pattern):
    features, labels = [], []

    for _, folder in _iter_tensor_folders(root_dir, pattern):
        tensor_paths = {
            emotion: os.path.join(folder, f"{emotion}.pt") for emotion in EMOTION_FILES
        }

        missing = [emotion for emotion, path in tensor_paths.items() if not os.path.exists(path)]
        if missing:
            print(f"Warning: missing tensors {missing} in {folder}; skipping")
            continue

        label_path = os.path.join(folder, "new_label.txt")
        if not os.path.isfile(label_path):
            print(f"Warning: missing label in {folder}; skipping")
            continue

        with open(label_path) as f:
            score = float(f.read().strip())

        label = 1 if score >= 53 else 0

        for emotion in EMOTION_FILES:
            feature = load_feature(tensor_paths[emotion])
            features.append(feature)
            labels.append(label)

    if not features:
        raise ValueError(f"No tensor samples found under {root_dir}")

    stacked_features = torch.stack(features)
    tensor_labels = torch.tensor(labels, dtype=torch.long)
    return dep_data(stacked_features, tensor_labels)
