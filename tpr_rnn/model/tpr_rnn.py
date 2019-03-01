from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from tpr_rnn.model.utils import MLP, LayerNorm, OptionalLayer


class TprRnn(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(TprRnn, self).__init__()
        # word embeddings
        self.word_embed = nn.Embedding(num_embeddings=config["vocab_size"],
                                       embedding_dim=config["symbol_size"])
        nn.init.uniform_(self.word_embed.weight, -config["init_limit"], config["init_limit"])
        # positional embeddings
        self.pos_embed = nn.Parameter(torch.ones(config["max_seq"], config["symbol_size"]))
        nn.init.ones_(self.pos_embed.data)
        self.pos_embed.data /= config["max_seq"]
        # output embeddings
        self.Z = nn.Parameter(torch.zeros(config["entity_size"], config["vocab_size"]))
        nn.init.xavier_uniform_(self.Z.data)

        self.update_module = UpdateModule(embed_layer=self.word_embed,
                                          embed_pos=self.pos_embed,
                                          config=config)
        self.inference_module = InferenceModule(embed_layer=self.word_embed,
                                                embed_pos=self.pos_embed,
                                                Z=self.Z,
                                                config=config)

    def forward(self, story: torch.Tensor, query: torch.Tensor):
        TPR = self.update_module(story)
        logits = self.inference_module(query, TPR)
        return logits


class UpdateModule(nn.Module):
    def __init__(self, embed_layer: nn.Embedding, embed_pos: nn.Parameter,
                 config: Dict[str, Any]):
        super(UpdateModule, self).__init__()
        self.embed_layer = embed_layer
        self.embed_pos = embed_pos
        self.role_size = config["role_size"]
        self.ent_size = config["entity_size"]
        self.hidden_size = config["hidden_size"]

        _, symbol_size = self.embed_layer.weight.size()

        self.e = nn.ModuleList([MLP(equation='bse,er->bsr', in_features=symbol_size,
                                    hidden_size=self.hidden_size, out_size=self.ent_size) for _ in range(2)])
        self.r = nn.ModuleList([MLP(equation='bse,er->bsr', in_features=symbol_size,
                                    hidden_size=self.hidden_size, out_size=self.role_size) for _ in range(3)])

    def forward(self, story: torch.Tensor) -> torch.Tensor:
        sentence_embed = self.embed_layer(story)  # [b, s, w, e]
        # TODO: check the validness of embeddings
        batch_size = sentence_embed.size(0)

        sentence_sum = torch.einsum('bswe,we->bse', sentence_embed, self.embed_pos)

        e1, e2 = [module(sentence_sum) for module in self.e]
        r1, r2, r3 = [module(sentence_sum) for module in self.r]
        partial_add_W = torch.einsum('bsr,bsf->bsrf', r1, e2)
        partial_add_B = torch.einsum('bsr,bsf->bsrf', r3, e1)

        inputs = (e1, r1, partial_add_W, e2, r2, partial_add_B, r3)

        # TPR-RNN steps
        TPR = torch.zeros(batch_size, self.ent_size, self.role_size, self.ent_size).to(story.device)
        for x in zip(*[torch.unbind(t, dim=1) for t in inputs]):
            e1_i, r1_i, partial_add_W_i, e2_i, r2_i, partial_add_B_i, r3_i = x
            w_hat = torch.einsum('be,br,berf->bf', e1_i, r1_i, TPR)
            partial_remove_W = torch.einsum('br,bf->brf', r1_i, w_hat)

            m_hat = torch.einsum('be,br,berf->bf', e1_i, r2_i, TPR)
            partial_remove_M = torch.einsum('br,bf->brf', r2_i, m_hat)
            partial_add_M = torch.einsum('br,bf->brf', r2_i, w_hat)

            b_hat = torch.einsum('be,br,berf->bf', e2_i, r3_i, TPR)
            partial_remove_B = torch.einsum('br,bf->brf', r3_i, b_hat)

            # operations
            write_op = partial_add_W_i - partial_remove_W
            move_op = partial_add_M - partial_remove_M
            backlink_op = partial_add_B_i - partial_remove_B
            delta_F = torch.einsum('be,brf->berf', e1_i, write_op + move_op) + \
                      torch.einsum('be,brf->berf', e2_i, backlink_op)
            TPR = TPR + delta_F
        return TPR


class InferenceModule(nn.Module):
    def __init__(self, embed_layer: nn.Embedding, embed_pos: nn.Parameter, Z: nn.Parameter,
                 config: Dict[str, Any]):
        super(InferenceModule, self).__init__()
        self.embed_layer = embed_layer
        self.embed_pos = embed_pos
        self.Z = Z
        self.hidden_size = config["hidden_size"]
        self.ent_size = config["entity_size"]
        self.role_size = config["role_size"]

        _, symbol_size = self.embed_layer.weight.size()

        self.e = nn.ModuleList([MLP(equation='be,er->br', in_features=symbol_size,
                                    hidden_size=self.hidden_size, out_size=self.ent_size) for _ in range(2)])
        self.r = nn.ModuleList([MLP(equation='be,er->br', in_features=symbol_size,
                                    hidden_size=self.hidden_size, out_size=self.role_size) for _ in range(3)])
        self.l1, self.l2, self.l3 = [OptionalLayer(LayerNorm(hidden_size=self.ent_size), active=config["LN"])
                                     for _ in range(3)]

    def forward(self, query: torch.Tensor, TPR: torch.Tensor):
        query_embed = self.embed_layer(query)  # [b, w, e]
        query_sum = torch.einsum('bwe,we->be', query_embed, self.embed_pos)

        e1, e2 = [module(query_sum) for module in self.e]
        r1, r2, r3 = [module(query_sum) for module in self.r]

        i1 = self.l1(torch.einsum('be,br,berf->bf', e1, r1, TPR))
        i2 = self.l2(torch.einsum('be,br,berf->bf', i1, r2, TPR))
        i3 = self.l3(torch.einsum('be,br,berf->bf', i2, r3, TPR))

        step_sum = i1 + i2 + i3
        logits = torch.einsum('bf,fl->bl', step_sum, self.Z.data)
        return logits
