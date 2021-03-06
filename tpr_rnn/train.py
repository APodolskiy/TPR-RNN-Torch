from argparse import ArgumentParser
import logging
import json
from pathlib import Path
import shutil
from tqdm import tqdm
from typing import Dict

import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from tpr_rnn.data_preprocess.preprocess import parse
from tpr_rnn.model.tpr_rnn import TprRnn
from tpr_rnn.model.utils import WarmupScheduler

logger = logging.getLogger(__name__)


def train(config: Dict[str, Dict],
          serialization_path: str,
          eval_test: bool = False,
          force: bool = True) -> None:
    # Create serialization dir
    dir_path = Path(serialization_path)
    if dir_path.exists() and force:
        shutil.rmtree(dir_path)
    dir_path.mkdir(parents=True, exist_ok=False)
    model_path = dir_path / "model.pt"
    config_path = dir_path / "config.json"
    writer = SummaryWriter(log_dir=str(dir_path))

    # Read config
    data_config = config["data"]
    trainer_config = config["trainer"]
    model_config = config["model"]
    optimizer_config = config["optimizer"]
    # Load data
    train_raw_data, valid_raw_data, test_raw_data, word2id = parse(data_config["data_path"], data_config["task-id"])
    id2word = {word2id[k]: k for k in word2id}

    train_epoch_size = train_raw_data[0].shape[0]
    valid_epoch_size = valid_raw_data[0].shape[0]
    test_epoch_size = test_raw_data[0].shape[0]
    vocab_size = len(word2id)

    max_story_length = np.max(train_raw_data[1])
    max_sentences = train_raw_data[0].shape[1]
    max_seq = train_raw_data[0].shape[2]
    max_q = train_raw_data[0].shape[1]
    valid_batch_size = valid_epoch_size // 73  # like in the original implementation
    test_batch_size = test_epoch_size // 73

    model_config["vocab_size"] = vocab_size
    model_config["max_seq"] = max_seq
    model_config["symbol_size"] = vocab_size

    train_dataset = TensorDataset(*[torch.LongTensor(a) for a in train_raw_data[:-1]])
    valid_dataset = TensorDataset(*[torch.LongTensor(a) for a in valid_raw_data[:-1]])

    train_data_loader = DataLoader(train_dataset, batch_size=trainer_config["batch_size"], shuffle=True)
    valid_data_loader = DataLoader(valid_dataset, batch_size=valid_batch_size)

    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TprRnn(model_config).to(device)
    optimizer = optim.Adam(model.parameters(),
                           lr=optimizer_config["lr"], betas=(optimizer_config["beta1"], optimizer_config["beta2"]))

    loss_fn = nn.CrossEntropyLoss(reduction='none')
    warm_up = optimizer_config.get("warm_up", False)

    scheduler = WarmupScheduler(optimizer=optimizer,
                                steps=optimizer_config["warm_up_steps"] if warm_up else 0,
                                multiplier=optimizer_config["warm_up_factor"] if warm_up else 1)

    decay_done = False
    for i in range(trainer_config["epochs"]):
        logging.info(f"##### EPOCH: {i} #####")
        scheduler.step()
        # Train
        model.train()
        correct = 0
        train_loss = 0
        for story, story_length, query, answer in tqdm(train_data_loader):
            optimizer.zero_grad()
            logits = model(story.to(device), query.to(device))
            answer = answer.to(device)
            correct_batch = (torch.argmax(logits, dim=-1) == answer).sum()
            correct += correct_batch.item()

            loss = loss_fn(logits, answer)
            train_loss += loss.sum().item()
            loss = loss.mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), optimizer_config["max_gradient_norm"])
            optimizer.step()
        train_acc = correct / train_epoch_size
        train_loss = train_loss / train_epoch_size

        # Validation
        model.eval()
        correct = 0
        valid_loss = 0
        with torch.no_grad():
            for story, story_length, query, answer in tqdm(valid_data_loader):
                logits = model(story.to(device), query.to(device))
                answer = answer.to(device)
                correct_batch = (torch.argmax(logits, dim=-1) == answer).sum()
                correct += correct_batch.item()
                loss = loss_fn(logits, answer)
                valid_loss += loss.sum().item()
            valid_acc = correct / valid_epoch_size
            valid_loss = valid_loss / valid_epoch_size

        writer.add_scalars("accuracy", {"train": train_acc,
                                        "validation": valid_acc}, i)
        writer.add_scalars("loss", {"train": train_loss,
                                    "validation": valid_loss}, i)

        logging.info(f"\nTrain accuracy: {train_acc:.3f}, loss: {train_loss:.3f}"
                     f"\nValid accuracy: {valid_acc:.3f}, loss: {valid_loss:.3f}")
        if optimizer_config.get("decay", False) and valid_loss < optimizer_config["decay_thr"] and not decay_done:
            scheduler.decay_lr(optimizer_config["decay_factor"])
            decay_done = True

    if eval_test:
        # Evaluation on test data
        test_dataset = TensorDataset(*[torch.LongTensor(a) for a in test_raw_data[:-1]])
        test_data_loader = DataLoader(test_dataset, batch_size=test_batch_size)

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
                test_loss += loss.sum().item()
            test_acc = correct / test_epoch_size
            test_loss = test_loss / test_epoch_size
        logging.info(f"Test accuracy: {test_acc:.3f}, loss: {test_loss:.3f}")

    torch.save(model.state_dict(), model_path.absolute())
    with config_path.open("w") as fp:
        json.dump(config, fp, indent=4)
    writer.close()


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script for TPR RNN.")
    parser.add_argument("--config-file", type=str, metavar='PATH', required=True,
                        help="Path to the model config file")
    parser.add_argument("--serialization-path", type=str, metavar='PATH', required=True,
                        help="Serialization directory path")
    parser.add_argument("--eval-test", default=False, action='store_true',
                        help="Whether to eval model on test dataset after training (default: False)")
    parser.add_argument("--logging-level", type=str, metavar='LEVEL', default=20, choices=range(10, 51, 10),
                        help="Logging level (default: 20)")
    args = parser.parse_args()

    logging.basicConfig(level=args.logging_level)

    with open(args.config_file, "r") as fp:
        config = json.load(fp)

    train(config, args.serialization_path, args.eval_test)
