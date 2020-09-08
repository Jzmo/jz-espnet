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
    if args.log is not None:
        with open(args.log) as f:
            logs = json.load(f)
        val_scores = []
        for log in logs:
            if args.metric == "acc":
                if "validation/main/acc" in log.keys():
                    val_scores += [[log["epoch"], log["validation/main/acc"]]]
            elif args.metric == "perplexity":
                if "val_perplexity" in log.keys():
                    val_scores += [[log["epoch"], 1 / log["val_perplexity"]]]
            elif args.metric == "loss":
                if "validation/main/loss" in log.keys():
                    val_scores += [[log["epoch"], -log["validation/main/loss"]]]
            elif args.metric == "bleu":
                if "validation/main/bleu" in log.keys():
                    val_scores += [[log["epoch"], log["validation/main/bleu"]]]
            elif args.metric == "cer_ctc":
                if "validation/main/cer_ctc" in log.keys():
                    val_scores += [[log["epoch"], -log["validation/main/cer_ctc"]]]
            else:
                # Keep original order for compatibility
                if "validation/main/acc" in log.keys():
                    val_scores += [[log["epoch"], log["validation/main/acc"]]]
                elif "val_perplexity" in log.keys():
                    val_scores += [[log["epoch"], 1 / log["val_perplexity"]]]
                elif "validation/main/loss" in log.keys():
                    val_scores += [[log["epoch"], -log["validation/main/loss"]]]

        if len(val_scores) == 0:
            raise ValueError("%s is not found in log." % args.metric)
        val_scores = np.array(val_scores)
        sort_idx = np.argsort(val_scores[:, -1])
        sorted_val_scores = val_scores[sort_idx][::-1]
        print("best val scores = ", np.max(sorted_val_scores[: args.num, 1]))

    import torch

    model, train_args = load_trained_model(args.out)
    print(model)
    print(
        "# params:",
        sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000,
    )
    print("===================================================")


def get_parser():
    parser = argparse.ArgumentParser(description="average models from snapshot")
    parser.add_argument("--snapshots", required=True, type=str, nargs="+")
    parser.add_argument("--out", required=True, type=str)
    parser.add_argument("--num", default=10, type=int)
    parser.add_argument("--backend", default="chainer", type=str)
    parser.add_argument("--log", default=None, type=str, nargs="?")
    parser.add_argument(
        "--metric",
        default="",
        type=str,
        nargs="?",
        choices=["acc", "bleu", "cer_ctc", "loss", "perplexity"],
    )
    return parser


def load_trained_model(model_path):
    """Load the trained model for recognition.
    Args:
        model_path (str): Path to model.***.best
    """
    idim, odim, train_args = get_model_conf(
        model_path, os.path.join(os.path.dirname(model_path), "model.json")
    )

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
