from argparse import ArgumentParser

from tpr_rnn.data_preprocess.preprocess import parse


def train():
    pass


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script for TPR RNN.")
    parser.add_argument("--data-path", type=str, required=True, metavar='PATH',
                        help="Path to the folder with training data")
    parser.add_argument("--task-id", type=str, required=True, metavar='ID',
                        help="Task id to train on")
    parser.add_argument("--config-file", type=str, metavar='PATH',
                        help="Path to the model config file")
    args = parser.parse_args()

    train_raw_data, valid_raw_data, _, word2id = parse(args.data_path, args.task_id)
    id2word = {word2id[k]: k for k in word2id}
    vocab_size = len(word2id)
