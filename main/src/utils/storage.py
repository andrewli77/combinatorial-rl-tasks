import csv
import os
import torch
import logging
import sys

import utils


def create_folders_if_necessary(path):
    dirname = os.path.dirname(path)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)


def get_storage_dir():
    if "RL_STORAGE" in os.environ:
        return os.environ["RL_STORAGE"]
    return "storage"


def get_model_dir(model_name):
    if "storage/" in model_name:
        return model_name
    else:
        return os.path.join(get_storage_dir(), model_name)


def get_model_dir(model_name, storage_dir="storage"):
    if storage_dir in model_name:
        return model_name
    return os.path.join(storage_dir, model_name)


def get_status_path(model_dir):
    return os.path.join(model_dir, "status.pt")


def get_status(model_dir):
    path = get_status_path(model_dir)
    return torch.load(path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))


def save_status(status, model_dir):
    path = get_status_path(model_dir)
    utils.create_folders_if_necessary(path)
    torch.save(status, path)


def get_vocab(model_dir):
    return get_status(model_dir)["vocab"]


def get_model_state(model_dir):
    return get_status(model_dir)["model_state"]

def get_hi_model_state(model_dir):
    return get_status(model_dir)["hi_model_state"]

def get_lo_model_state(model_dir):
    return get_status(model_dir)["lo_model_state"]

def get_dynamics_model_state(model_dir):
    return get_status(model_dir)["dynamics_model"]

def get_encoder_model_state(model_dir):
    return get_status(model_dir)["encoder_model"]

def get_txt_logger(model_dir):
    path = os.path.join(model_dir, "log.txt")
    utils.create_folders_if_necessary(path)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(filename=path),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logging.getLogger()


def get_csv_logger(model_dir):
    csv_path = os.path.join(model_dir, "log.csv")
    utils.create_folders_if_necessary(csv_path)
    csv_file = open(csv_path, "a")
    return csv_file, csv.writer(csv_file)
