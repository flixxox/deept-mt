import math

import torch
import torch.nn as nn
from numpy import dtype

from deept.util.globals import Context
from deept.model.state import DynamicState
from deept.model.model import register_model
from deept.model.modules import (
    SinusodialPositionalEmbedding,
    PositionalEmbedding
)

@register_model("TorchTransformer")
class Transformer(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.encoder = TransformerEncoder(**kwargs)
        self.decoder = TransformerDecoder(**kwargs)

        if self.tiew:
            self.decoder.tgt_embed.word_embed.weight = self.encoder.src_embed.word_embed.weight
            self.decoder.output_projection.weight = self.encoder.src_embed.word_embed.weight

    @staticmethod
    def create_from_config(config):
        
        model =  Transformer(
            pad_index = Context['vocab_src'].PAD,
            srcV = Context['vocab_src'].vocab_size,
            tgtV = Context['vocab_tgt'].vocab_size,
            input_keys = config['model_input'],
            encL = config['encL'],
            decL = config['decL'],
            model_dim = config['model_dim'],
            nHeads = config['nHeads'],
            ff_dim = config['ff_dim'],
            dropout = config['dropout'],
            maxI = config['max_sample_size'],
            tiew = config['tiew'],
            initializer = config['initializer'],
            variance_scaling_scale = config['variance_scaling_scale'],
            stepwise = config['stepwise'],
            use_sinusodial_pos_embed = config['use_sinusodial_pos_embed', True]
        )

        return model

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def __call__(self, src, tgt):

        masks = self.create_masks(src, tgt)
        
        h = self.encoder(src, **masks)

        s = self.decoder(tgt, h, **masks)

        return s, h

    def create_masks(self, src, tgt):

        src_mask = (src == self.pad_index).to(torch.bool)

        if tgt is not None:

            tgtT = tgt.shape[1]
            tgt_mask = torch.tril(tgt.new_ones((tgtT, tgtT)))
            tgt_mask = (tgt_mask == 0).to(torch.bool)

            return {'src_mask': src_mask,
                'tgt_mask': tgt_mask}
        else:
            return {'src_mask': src_mask}


class TransformerEncoder(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        for k, v in kwargs.items():
            setattr(self, k, v)

        if self.use_sinusodial_pos_embed:
            self.src_embed = SinusodialPositionalEmbedding(
                self.srcV, self.model_dim, self.maxI, self.dropout, self.pad_index
            )
        else:
            self.src_embed = PositionalEmbedding(
                self.srcV, self.model_dim, self.maxI, self.dropout, self.pad_index
            )
        
        self.layers = nn.ModuleList([TransformerEncoderLayer(**kwargs) for n in range(self.encL)])

        self.lnorm = nn.LayerNorm((self.model_dim))

    def __call__(self, src, src_mask=None, tgt_mask=None):

        h = self.src_embed(src)

        for layer in self.layers:

            h, _ = layer(h, src_mask=src_mask)

        h = self.lnorm(h)

        return h


class TransformerDecoder(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        for k, v in kwargs.items():
            setattr(self, k, v)

        if self.use_sinusodial_pos_embed:
            self.tgt_embed = SinusodialPositionalEmbedding(
                self.tgtV, self.model_dim, self.maxI, self.dropout, self.pad_index
            )
        else:
            self.tgt_embed = PositionalEmbedding(
                self.tgtV, self.model_dim, self.maxI, self.dropout, self.pad_index
            )

        self.layers = nn.ModuleList([TransformerDecoderLayer(**kwargs) for n in range(self.decL)])

        self.lnorm = nn.LayerNorm((self.model_dim))
        self.output_projection = nn.Linear(self.model_dim, self.tgtV)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def __call__(self, tgt, h, i=None, src_mask=None, tgt_mask=None):

        s = self.tgt_embed(tgt, J=i)

        for layer in self.layers:

            s, _, _ = layer(s, h, src_mask=src_mask, tgt_mask=tgt_mask)

        s = self.lnorm(s)
        s = self.output_projection(s)
        s = self.log_softmax(s)

        return s


class TransformerEncoderLayer(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        for k, v in kwargs.items():
            setattr(self, k, v)
        
        self.lnorm1 = nn.LayerNorm((self.model_dim))
        self.att = nn.MultiheadAttention(
            self.model_dim, self.nHeads,
            dropout=self.dropout,
            bias=True,
            add_bias_kv=False,
            add_zero_attn=False,
            kdim=None, vdim=None,
            batch_first=True
        )
        self.dropout = nn.Dropout(self.dropout)

        self.lnorm2 = nn.LayerNorm((self.model_dim))
        self.ff1 = nn.Linear(self.model_dim, self.ff_dim)
        self.relu = nn.ReLU()
        self.ff2 = nn.Linear(self.ff_dim, self.model_dim)

    def __call__(self, x, src_mask=None):

        r = x
        x = self.lnorm1(x)
        x, a = self.att(x, x, x,
            key_padding_mask=src_mask
        )
        x = self.dropout(x)
        x = x + r

        r = x
        x = self.lnorm2(x)
        x = self.ff1(x)
        x = self.relu(x)
        x = self.ff2(x)
        x = self.dropout(x)
        x = x + r

        return x, a


class TransformerDecoderLayer(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.lnorm1 = nn.LayerNorm((self.model_dim))
        self.self_att = nn.MultiheadAttention(
            self.model_dim, self.nHeads,
            dropout=self.dropout,
            bias=True,
            add_bias_kv=False,
            add_zero_attn=False,
            kdim=None, vdim=None,
            batch_first=True
        )
        self.self_att_state = DynamicState(time_dim=1, stepwise=self.stepwise)

        self.lnorm2 = nn.LayerNorm((self.model_dim))
        self.cross_att = nn.MultiheadAttention(
            self.model_dim, self.nHeads,
            dropout=self.dropout,
            bias=True,
            add_bias_kv=False,
            add_zero_attn=False,
            kdim=None, vdim=None,
            batch_first=True
        )

        self.lnorm3 = nn.LayerNorm((self.model_dim))
        self.ff1 = nn.Linear(self.model_dim, self.ff_dim)
        self.relu = nn.ReLU()
        self.ff2 = nn.Linear(self.ff_dim, self.model_dim)
        self.dropout = nn.Dropout(self.dropout)

    def __call__(self, s, h, src_mask=None, tgt_mask=None):

        r = s
        s = self.lnorm1(s)
        s_full = self.self_att_state.full(s)
        s, b = self.self_att(s, s_full, s_full,
            attn_mask=tgt_mask,
            is_causal=True
        )
        s = self.dropout(s)
        s = s + r

        r = s
        s = self.lnorm2(s)
        s, c = self.cross_att(s, h, h,
            key_padding_mask=src_mask
        )
        s = self.dropout(s)
        s = s + r

        r = s
        s = self.lnorm3(s)
        s = self.ff1(s)
        s = self.relu(s)
        s = self.ff2(s)
        s = self.dropout(s)
        s = s + r

        return s, b, c