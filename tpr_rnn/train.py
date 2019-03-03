from argparse import ArgumentParser

from tpr_rnn.data_preprocess.preprocess import parse


def train(config: Dict) -> None:
    data_config = config["data"]
    trainer_config = config["trainer"]
    model_config = config["model"]
    optimizer_config = config["optimizer"]
    # Load data
    train_raw_data, valid_raw_data, _, word2id = parse(data_config["data_path"], data_config["task-id"])
    id2word = {word2id[k]: k for k in word2id}
    vocab_size = len(word2id)
    model_config["vocab_size"] = vocab_size
    # Create model
    model = TprRnn(**model_config)


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script for TPR RNN.")
    parser.add_argument("--config-file", type=str, metavar='PATH', required=True,
                        help="Path to the model config file")
    args = parser.parse_args()

    train_raw_data, valid_raw_data, _, word2id = parse(args.data_path, args.task_id)
    id2word = {word2id[k]: k for k in word2id}
    vocab_size = len(word2id)
