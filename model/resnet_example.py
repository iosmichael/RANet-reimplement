import torch.nn as nn
import torch
from .resnet101 import ResNet101

def main():
    model = ResNet101(pretrained=True)
    x = torch.ones((1, 3, 224, 224))
    # y = model.forward_full(x)
    print("Resnet 101 Layer Output Description:")
    y = model.forward_partial(x, levels=4)
    print(f"# {len(y)} output features from the forward method")
    for i, layer in enumerate(y):
        print(f"# {i} size: {layer.size()}")

if __name__ == "__main__":
    main()
    pass
