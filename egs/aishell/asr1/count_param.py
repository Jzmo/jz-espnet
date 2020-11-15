#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from espnet.asr.asr_utils import get_model_conf
from espnet.asr.asr_utils import torch_load
from espnet.utils.dynamic_import import dynamic_import
import numpy as np


def main():
    if args.backend == "pytorch":
        model, train_args = load_trained_model(args.out)
        print(
            sum(p.numel() for name, p in model.named_parameters() if p.requires_grad)
            / 1000000
        )


def get_parser():
    parser = argparse.ArgumentParser(description="average models from snapshot")
    parser.add_argument("--out", required=True, type=str)
    return parser


def load_trained_model(model_path):
    print("reading model parameters from " + model_path)

    if hasattr(train_args, "model_module"):
        model_module = train_args.model_module
    else:
        model_module = "espnet.nets.pytorch_backend.e2e_asr:E2E"
    model_class = dynamic_import(model_module)
    model = model_class(idim, odim, train_args)

    torch_load(model_path, model)

    return model, train_args


if __name__ == "__main__":
    args = get_parser().parse_args()
    main()
