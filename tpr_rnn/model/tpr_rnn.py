import torch
import torch.nn as nn

from tpr_rnn.model.utils import MLP


class Model(nn.Module):
    def __init__(self, vocab_size: int, symbol_size: int, entity_size,
                 max_seq: int, hidden_size: int, out_size: int, role_size: int):
        super(Model, self).__init__()
        # word embeddings
        self.word_embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=symbol_size)
        nn.init.uniform_(self.word_embed.weight, -0.01, 0.01)
        # positional embeddings
        #self.pos_embed = nn.Embedding(num_embeddings=max_seq, embedding_dim=symbol_size)
        self.pos_embed = nn.Parameter(torch.ones(max_seq, symbol_size))
        nn.init.ones_(self.pos_embed.weight)
        self.pos_embed.weight.data /= max_seq
        # output embeddings
        self.Z = nn.Embedding(num_embeddings=entity_size, embedding_dim=vocab_size)
        nn.init.xavier_uniform_(self.Z.weight)

        self.update_module = UpdateModule(embed_layer=self.word_embed, embed_pos=self.pos_embed,
                                          hidden_size=hidden_size, out_size=out_size, role_size=role_size)
        self.inference_module = InferenceModule(embed_layer=self.word_embed)

    def forward(self, x: torch.Tensor):
        pass


class UpdateModule(nn.Module):
    def __init__(self, embed_layer: nn.Embedding, embed_pos: nn.Parameter,
                 hidden_size: int, out_size: int, role_size: int):
        super(UpdateModule, self).__init__()
        self.embed_layer = embed_layer
        self.embed_pos = embed_pos

        _, symbol_size = self.embed_layer.weight.size()

        self.e = nn.ModuleList([MLP(equation='bse,er->bsr', in_features=symbol_size,
                                    hidden_size=hidden_size, out_size=out_size) for _ in range(2)])
        self.r = nn.ModuleList([MLP(equation='bse,er->bsr', in_features=symbol_size,
                                    hidden_size=hidden_size, out_size=role_size) for _ in range(3)])

    def forward(self, story: torch.Tensor, query: torch.Tensor):
        sentence_embed = self.embed_layer(story)  # [b, s, w, e]
        query_embed = self.embed_layer(query)  # [b, w, e]

        sentence_sum = torch.einsum('bswe,we->bse', [sentence_embed, self.embed_pos])
        query_sum = torch.einsum('bwe,we->be', [query_embed, self.embed_pos])

        e1, e2 = [module(sentence_sum) for module in self.e]
        r1, r2, r3 = [module(sentence_sum) for module in self.r]
        partial_add_W = torch.einsum('bsr,bsf->bsrf', [r1, e2])
        partial_add_B = torch.einsum('bsr,bsf->bsrf', [r3, e1])

        inputs = (e1, r1, partial_add_W, e2, r2, partial_add_B, r3)
        pass


class InferenceModule(nn.Module):
    def __init__(self, embed_layer: nn.Module):
        super(InferenceModule, self).__init__()
        self.embed_layer = embed_layer

    def forward(self, x):
        pass
