from argparse import ArgumentParser
import json
from pathlib import Path
from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from seaborn.matrix import ClusterGrid
from sklearn.metrics.pairwise import cosine_similarity

import torch
from torch.utils.data import DataLoader, TensorDataset

from tpr_rnn.data_preprocess.preprocess import parse
from tpr_rnn.model.tpr_rnn import TprRnn


def cluster_analysis(dir_path: str, stories_num: int = 5):
    dir_path = Path(dir_path)
    # load config
    config_path = dir_path / "config.json"
    with open(config_path, "r") as fp:
        config = json.load(fp)
    data_config = config["data"]
    trainer_config = config["trainer"]
    model_config = config["model"]

    model_path = dir_path / "model.pt"
    model = TprRnn(model_config)
    model.load_state_dict(torch.load(model_path.absolute()))

    _, _, test_raw_data, word2id = parse(data_config["data_path"], data_config["task-id"])
    id2word = {word2id[k]: k for k in word2id}

    plot_small_random_sample(model, test_raw_data, id2word, dir_path, elem='e1', stories_num=stories_num)
    plot_small_random_sample(model, test_raw_data, id2word, dir_path, elem='e2', stories_num=stories_num)
    plot_small_random_sample(model, test_raw_data, id2word, dir_path, elem='r1', stories_num=stories_num)
    plot_small_random_sample(model, test_raw_data, id2word, dir_path, elem='r2', stories_num=stories_num)
    plot_small_random_sample(model, test_raw_data, id2word, dir_path, elem='r3', stories_num=stories_num)


def cluster(model: torch.nn.Module, test_raw_data: List,
            elem='e1', stories_num: int = 1000) -> Tuple[ClusterGrid, List]:
    model.eval()
    test_epoch_size = test_raw_data[0].shape[0]
    idxs = np.random.randint(low=1, high=test_epoch_size, size=stories_num)
    batch = [torch.LongTensor(test_raw_data[0][idxs, :, :]),
             torch.LongTensor(test_raw_data[1][idxs]),
             torch.LongTensor(test_raw_data[2][idxs, :]),
             torch.LongTensor(test_raw_data[3][idxs])]

    story, query = batch[0], batch[2]
    with torch.no_grad():
        res = model.get_elem(story=story, query=query, elem=elem).numpy()
        res = np.reshape(res, (-1, res.shape[-1]))
        story = story.numpy()
        sentences = np.reshape(story, (-1, story.shape[-1]))
        _, indecies = np.unique(sentences, axis=0, return_index=True)
        sentences = sentences[indecies]
        r = res[indecies]
        C = cosine_similarity(r)
        g = sns.clustermap(C, standard_scale=1, figsize=(20, 20))
        return g, sentences


def plot_small_random_sample(model: torch.nn.Module, test_raw_data: List,
                             id2word: Dict, save_dir: str,
                             elem: str, stories_num: int = 5,) -> None:
    save_path = Path(save_dir)
    g, sentences = cluster(model, test_raw_data, elem, stories_num)
    g.savefig(str(save_path / f"small_plot_{elem}.png"))
    for idx in g.dendrogram_row.reordered_ind:
        print("{:4}: {}".format(idx, translate(sentences[idx], id2word)))


def translate(arr: np.ndarray, id2word: Dict):
    assert (type(arr) == np.ndarray), "You can only translate numpy arrays"
    old_shape = arr.shape
    arr = np.reshape(arr, (-1))
    arr = np.asarray([id2word[x] for x in arr])
    arr = np.reshape(arr, old_shape)
    as_string = np.apply_along_axis(lambda x: " ".join(x), axis=-1, arr=arr)
    return as_string


if __name__ == "__main__":
    parser = ArgumentParser(description="Cluster analysis of pre-trained model")
    parser.add_argument("--model-path", type=str, required=True, metavar='PATH',
                        help="Path to pre-trained model directory.")
    parser.add_argument("--num-stories", type=int, required=False, metavar='N',
                        default=5, help="Number of stories to cluster on (default: 5)")
    args = parser.parse_args()

    cluster_analysis(args.model_path, args.num_stories)
