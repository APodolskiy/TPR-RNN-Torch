import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, vocab_size: int, symbol_size: int, entity_size, max_seq: int):
        super(Model, self).__init__()
        # word embeddings
        self.word_embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=symbol_size)
        nn.init.uniform_(self.word_embed.weight, -0.01, 0.01)
        # positional embeddings
        self.pos_embed = nn.Embedding(num_embeddings=max_seq, embedding_dim=symbol_size)
        nn.init.ones_(self.pos_embed.weight)
        self.pos_embed.weight.data /= max_seq
        # output embeddings
        self.Z = nn.Embedding(num_embeddings=entity_size, embedding_dim=vocab_size)
        nn.init.xavier_uniform_(self.Z.weight)

        self.update_module = UpdateModule(embed_layer=self.word_embed)
        self.inference_module = InferenceModule(embed_layer=self.word_embed)

    def forward(self, x: torch.Tensor):
        pass


class UpdateModule(nn.Module):
    def __init__(self, embed_layer: nn.Module):
        super(UpdateModule, self).__init__()
        self.embed_layer = embed_layer

    def forward(self, x):
        pass


class InferenceModule(nn.Module):
    def __init__(self, embed_layer: nn.Module):
        super(InferenceModule, self).__init__()
        self.embed_layer = embed_layer

    def forward(self, x):
        pass
