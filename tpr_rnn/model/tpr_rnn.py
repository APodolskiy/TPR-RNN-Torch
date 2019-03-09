from typing import Dict, Any

import torch
import torch.nn as nn

from tpr_rnn.model.utils import MLP, LayerNorm, OptionalLayer


class TprRnn(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(TprRnn, self).__init__()
        self.input_module = InputModule(config)
        self.update_module = UpdateModule(config=config)
        self.inference_module = InferenceModule(config=config)

    def forward(self, story: torch.Tensor, query: torch.Tensor):
        story_embed, query_embed = self.input_module(story, query)
        TPR = self.update_module(story_embed)
        logits = self.inference_module(query_embed, TPR)
        return logits


class InputModule(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(InputModule, self).__init__()
        self.word_embed = nn.Embedding(num_embeddings=config["vocab_size"],
                                       embedding_dim=config["symbol_size"])
        nn.init.uniform_(self.word_embed.weight, -config["init_limit"], config["init_limit"])
        # positional embeddings
        self.pos_embed = nn.Parameter(torch.ones(config["max_seq"], config["symbol_size"]))
        nn.init.ones_(self.pos_embed.data)
        self.pos_embed.data /= config["max_seq"]

    def forward(self, story: torch.Tensor, query: torch.Tensor):
        # Sentence embedding
        sentence_embed = self.word_embed(story)  # [b, s, w, e]
        sentence_sum = torch.einsum('bswe,we->bse', sentence_embed, self.pos_embed)
        # Query embedding
        query_embed = self.word_embed(query)  # [b, w, e]
        query_sum = torch.einsum('bwe,we->be', query_embed, self.pos_embed)
        return sentence_sum, query_sum


class UpdateModule(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(UpdateModule, self).__init__()
        self.role_size = config["role_size"]
        self.ent_size = config["entity_size"]
        self.hidden_size = config["hidden_size"]
        self.symbol_size = config["symbol_size"]

        self.e = nn.ModuleList([MLP(equation='bse,er->bsr', in_features=self.symbol_size,
                                    hidden_size=self.hidden_size, out_size=self.ent_size) for _ in range(2)])
        self.r = nn.ModuleList([MLP(equation='bse,er->bsr', in_features=self.symbol_size,
                                    hidden_size=self.hidden_size, out_size=self.role_size) for _ in range(3)])

    def forward(self, sentence_embed: torch.Tensor) -> torch.Tensor:
        batch_size = sentence_embed.size(0)

        e1, e2 = [module(sentence_embed) for module in self.e]
        r1, r2, r3 = [module(sentence_embed) for module in self.r]
        partial_add_W = torch.einsum('bsr,bsf->bsrf', r1, e2)
        partial_add_B = torch.einsum('bsr,bsf->bsrf', r3, e1)

        inputs = (e1, r1, partial_add_W, e2, r2, partial_add_B, r3)

        # TPR-RNN steps
        TPR = torch.zeros(batch_size, self.ent_size, self.role_size, self.ent_size).to(sentence_embed.device)
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
    def __init__(self, config: Dict[str, Any]):
        super(InferenceModule, self).__init__()
        self.hidden_size = config["hidden_size"]
        self.ent_size = config["entity_size"]
        self.role_size = config["role_size"]
        self.symbol_size = config["symbol_size"]
        # output embeddings
        self.Z = nn.Parameter(torch.zeros(config["entity_size"], config["vocab_size"]))
        nn.init.xavier_uniform_(self.Z.data)

        # TODO: remove unused entity head?
        self.e = nn.ModuleList([MLP(equation='be,er->br', in_features=self.symbol_size,
                                    hidden_size=self.hidden_size, out_size=self.ent_size) for _ in range(2)])
        self.r = nn.ModuleList([MLP(equation='be,er->br', in_features=self.symbol_size,
                                    hidden_size=self.hidden_size, out_size=self.role_size) for _ in range(3)])
        self.l1, self.l2, self.l3 = [OptionalLayer(LayerNorm(hidden_size=self.ent_size), active=config["LN"])
                                     for _ in range(3)]

    def forward(self, query_embed: torch.Tensor, TPR: torch.Tensor):
        e1, e2 = [module(query_embed) for module in self.e]
        r1, r2, r3 = [module(query_embed) for module in self.r]

        i1 = self.l1(torch.einsum('be,br,berf->bf', e1, r1, TPR))
        i2 = self.l2(torch.einsum('be,br,berf->bf', i1, r2, TPR))
        i3 = self.l3(torch.einsum('be,br,berf->bf', i2, r3, TPR))

        step_sum = i1 + i2 + i3
        logits = torch.einsum('bf,fl->bl', step_sum, self.Z.data)
        return logits
