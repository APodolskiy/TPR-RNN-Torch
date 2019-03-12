from argparse import ArgumentParser
import logging
import json
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from tpr_rnn.model.tpr_rnn import TprRnn
from tpr_rnn.data_preprocess.preprocess import parse


def evaluate(model_dir: str, no_cuda: bool = True):
    dir_path = Path(model_dir)

    config_path = dir_path / "config.json"
    with open(config_path, "r") as fp:
        config = json.load(fp)
    data_config = config["data"]
    trainer_config = config["trainer"]
    model_config = config["model"]

    _, _, test_raw_data, word2id = parse(data_config["data_path"], data_config["task-id"])
    id2word = {word2id[k]: k for k in word2id}

    model_path = dir_path / "model.pt"
    model = TprRnn(model_config)
    model.load_state_dict(torch.load(model_path.absolute()))

    test_epoch_size = test_raw_data[0].shape[0]
    test_batch_size = test_epoch_size // 73
    test_dataset = TensorDataset(*[torch.LongTensor(a) for a in test_raw_data[:-1]])
    test_data_loader = DataLoader(test_dataset, batch_size=test_batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else 'cpu')
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss(reduction='sum')
    model.eval()
    correct = 0
    test_loss = 0
    with torch.no_grad():
        for story, story_length, query, answer in tqdm(test_data_loader):
            logits = model(story.to(device), query.to(device))
            answer = answer.to(device)
            correct_batch = (torch.argmax(logits, dim=-1) == answer).sum()
            correct += correct_batch.item()
            loss = loss_fn(logits, answer)
            test_loss += loss.item()
        test_acc = correct / test_epoch_size
        test_loss = test_loss / test_epoch_size
    print(f"Test accuracy: {test_acc:.3f}, loss: {test_loss:.3f}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Pre-trained model evaluation script")
    parser.add_argument("--model-dir", type=str, required=True, metavar='PATH',
                        help="Path to the folder with pre-trained model")
    parser.add_argument("--no-cuda", action="store_true",
                        help="Not to use cuda for evaluation")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    evaluate(args.model_dir, no_cuda=args.no_cuda or True)
