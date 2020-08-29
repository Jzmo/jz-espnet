import argparse
import json
import os

import numpy as np
import torch
from torchsummary import summary


def main(args):
    print(args.path)

    states = torch.load(args.path, map_location=torch.device("cpu"))
    summary(states, (1028, 256, 256))
    raise ValueError
    for name, param in states.named_parameters():
        if param.requires_grad:
            print(name, param.data)


def get_parser():
    parser = argparse.ArgumentParser(description="average models from snapshot")
    parser.add_argument("--path", required=True, type=str)
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
