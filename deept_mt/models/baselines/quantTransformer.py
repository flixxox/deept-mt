import math

import torch
import torch.nn as nn

from deept.util.debug import my_print
from deept.model.state import DynamicState
from deept.model.model import register_model
from deept.util.globals import Context, Settings
from deept.model.modules import (
    LayerNormalization,
    PositionalEmbedding,
    SinusodialPositionalEmbedding
)


from deept_mt.quantization.quant_utils import get_dtype_from_string
from deept_mt.quantization.quantizer import (
    FakeWeightQuantizer,
    FakeActivationQuantizer
)
from deept_mt.quantization.quantized_modules import (
    QuantizedLinear,
    QuantizedLinearRelu
)

@register_model("QuantTransformer")
class QuantTransformer(nn.Module):

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
        
        model =  QuantTransformer(
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
            use_sinusodial_pos_embed = config['use_sinusodial_pos_embed', True],
            # -- QuantTransformer
            weight_quant_dtype = config['weight_quant_dtype'],
            weight_quant_method = config['weight_quant_method'],
            activation_quant_dtype = config['activation_quant_dtype'],
            activation_quant_method = config['activation_quant_method'],
            dot_quant_dtype = config['dot_quant_dtype'],
            dot_quant_method = config['dot_quant_method'],
            Av_quant_dtype = config['Av_quant_dtype'],
            Av_quant_method = config['Av_quant_method'],
            bits_others = config['bits_others'],
            bits_Wq = config['bits_Wq'],
            bits_Wk = config['bits_Wk'],
            bits_Wv = config['bits_Wv'],
            bits_dot = config['bits_dot'],
            bits_Av = config['bits_Av'],
            bits_Wo = config['bits_Wo'],
        )

        return model

    def init_weights(self):

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        size_bytes = self.calculate_model_size_in_bytes()

        my_print(f'Model size: {(size_bytes/1e6):6.1f}MBit!')

    def init_weights_from_checkpoint(self, checkpoint_path):
        
        checkpoint = torch.load(checkpoint_path, map_location=Settings.get_device())

        for k, v in self.state_dict().items():
            if k not in checkpoint['model']:
                if 'observer' not in k:
                    raise AssertionError(f"""Error loading weights from checkpoint.
                        Only observer variables should be missing but did not find {k}!
                    """)

        self.load_state_dict(checkpoint['model'], strict=False)
    
    def __call__(self, src, tgt):

        masks = self.create_masks(src, tgt)
        
        h = self.encoder(src, **masks)

        s = self.decoder(tgt, h, **masks)

        return s, h

    def create_masks(self, src, tgt):

        src_mask = (src == self.pad_index)
        src_mask = src_mask.unsqueeze(1).unsqueeze(2)

        if tgt is not None:

            tgtT = tgt.shape[1]
            tgt_mask = torch.tril(tgt.new_ones((tgtT, tgtT)))
            tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(1)  # B, H, I, I
            tgt_mask = (tgt_mask == 0)

            return {'src_mask': src_mask,
                'tgt_mask': tgt_mask}
        else:
            return {'src_mask': src_mask}

    def calculate_model_size_in_bytes(self):

        size_bytes = 0
        number_of_params = 0
        for name, p in self.named_parameters():
            number_of_params += p.numel()
            if 'weight' in name:
                if 'ff1' in name or 'ff2' in name:
                    size_bytes += (p.numel()*self.bits_others)
                elif 'W_q' in name:
                    size_bytes += (p.numel()*self.bits_Wq)
                elif 'W_k' in name:
                    size_bytes += (p.numel()*self.bits_Wk)
                elif 'W_v' in name:
                    size_bytes += (p.numel()*self.bits_Wv)
                elif 'W_o' in name:
                    size_bytes += (p.numel()*self.bits_Wo)
                else:
                    size_bytes += (p.numel()*32)
            else:
                size_bytes += (p.numel()*32)

        return size_bytes



class TransformerEncoder(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        for k, v in kwargs.items():
            setattr(self, k, v)

        if self.use_sinusodial_pos_embed:
            self.src_embed = SinusodialPositionalEmbedding(self.srcV, self.model_dim, self.maxI, self.dropout, self.pad_index)
        else:
            self.src_embed = PositionalEmbedding(self.srcV, self.model_dim, self.maxI, self.dropout, self.pad_index)
        
        self.layers = nn.ModuleList([TransformerEncoderLayer(**kwargs) for n in range(self.encL)])

        self.lnorm = LayerNormalization(self.model_dim)

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
            self.tgt_embed = SinusodialPositionalEmbedding(self.tgtV, self.model_dim, self.maxI, self.dropout, self.pad_index)
        else:
            self.tgt_embed = PositionalEmbedding(self.tgtV, self.model_dim, self.maxI, self.dropout, self.pad_index)

        self.layers = nn.ModuleList([TransformerDecoderLayer(**kwargs) for n in range(self.decL)])

        self.lnorm              = LayerNormalization(self.model_dim)
        self.output_projection  = nn.Linear(self.model_dim, self.tgtV)
        self.log_softmax        = nn.LogSoftmax(dim=-1)

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
        
        self.lnorm1 = LayerNormalization(self.model_dim)
        self.att = self.__create_multi_head_attention()
        
        self.lnorm2 = LayerNormalization(self.model_dim)

        if self.bits_others < 16:
            self.ff1 = QuantizedLinearRelu(self.model_dim, self.ff_dim, 
                bits=self.bits_others,
                weight_quant_dtype=self.weight_quant_dtype,
                weight_quant_method=self.weight_quant_method,
                activation_quant_dtype=self.activation_quant_dtype,
                activation_quant_method=self.activation_quant_method
            )
        else:
            self.ff1 = nn.Linear(self.model_dim, self.ff_dim)
            self.relu = nn.ReLU()

        if self.bits_others < 16:
            self.ff2 = QuantizedLinear(self.ff_dim, self.model_dim, 
                bits=self.bits_others,
                weight_quant_dtype=self.weight_quant_dtype,
                weight_quant_method=self.weight_quant_method,
                activation_quant_dtype=self.activation_quant_dtype,
                activation_quant_method=self.activation_quant_method
            )
        else:
            self.ff2 = nn.Linear(self.ff_dim, self.model_dim)
        self.dropout = nn.Dropout(self.dropout)

    def __create_multi_head_attention(self):
        return QuantizedMultiHeadAttention(self.nHeads, self.model_dim, self.dropout,
            weight_quant_dtype = self.weight_quant_dtype,
            weight_quant_method = self.weight_quant_method,
            activation_quant_dtype = self.activation_quant_dtype,
            activation_quant_method = self.activation_quant_method,
            dot_quant_dtype = self.dot_quant_dtype,
            dot_quant_method = self.dot_quant_method,
            Av_quant_dtype = self.Av_quant_dtype,
            Av_quant_method = self.Av_quant_method,
            bits_Wq = self.bits_Wq,
            bits_Wk = self.bits_Wk,
            bits_Wv = self.bits_Wv,
            bits_dot = self.bits_dot,
            bits_Av = self.bits_Av,
            bits_Wo = self.bits_Wo
        )

    def __call__(self, x, src_mask=None):

        r = x
        x = self.lnorm1(x)
        x, a = self.att(x, x, x, m=src_mask)
        x = self.dropout(x)
        x = x + r

        r = x
        x = self.lnorm2(x)
        x = self.ff1(x)
        if self.bits_others >= 16:
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

        self.lnorm1 = LayerNormalization(self.model_dim)
        self.self_att = self.__create_multi_head_attention()
        self.self_att_state = DynamicState(time_dim=1, stepwise=self.stepwise)

        self.lnorm2 = LayerNormalization(self.model_dim)
        self.cross_att = self.__create_multi_head_attention()

        self.lnorm3 = LayerNormalization(self.model_dim)
        if self.bits_others < 16:
            self.ff1 = QuantizedLinearRelu(self.model_dim, self.ff_dim, 
                bits=self.bits_others,
                weight_quant_dtype=self.weight_quant_dtype,
                weight_quant_method=self.weight_quant_method,
                activation_quant_dtype=self.activation_quant_dtype,
                activation_quant_method=self.activation_quant_method
            )
        else:
            self.ff1 = nn.Linear(self.model_dim, self.ff_dim)
            self.relu = nn.ReLU()

        if self.bits_others < 16:
            self.ff2 = QuantizedLinear(self.ff_dim, self.model_dim,
                bits=self.bits_others,
                weight_quant_dtype=self.weight_quant_dtype,
                weight_quant_method=self.weight_quant_method,
                activation_quant_dtype=self.activation_quant_dtype,
                activation_quant_method=self.activation_quant_method
            )
        else:
            self.ff2 = nn.Linear(self.ff_dim, self.model_dim)

        self.dropout = nn.Dropout(self.dropout)

    def __create_multi_head_attention(self):
        return QuantizedMultiHeadAttention(self.nHeads, self.model_dim, self.dropout,
            weight_quant_dtype = self.weight_quant_dtype,
            weight_quant_method = self.weight_quant_method,
            activation_quant_dtype = self.activation_quant_dtype,
            activation_quant_method = self.activation_quant_method,
            dot_quant_dtype = self.dot_quant_dtype,
            dot_quant_method = self.dot_quant_method,
            Av_quant_dtype = self.Av_quant_dtype,
            Av_quant_method = self.Av_quant_method,
            bits_Wq = self.bits_Wq,
            bits_Wk = self.bits_Wk,
            bits_Wv = self.bits_Wv,
            bits_dot = self.bits_dot,
            bits_Av = self.bits_Av,
            bits_Wo = self.bits_Wo
        )

    def __call__(self, s, h, src_mask=None, tgt_mask=None):

        r = s
        s = self.lnorm1(s)
        s_full = self.self_att_state.full(s)
        s, b = self.self_att(s, s_full, s_full, m=tgt_mask)
        s = self.dropout(s)
        s = s + r

        r = s
        s = self.lnorm2(s)
        s, c = self.cross_att(s, h, h, m=src_mask)
        s = self.dropout(s)
        s = s + r

        r = s
        s = self.lnorm3(s)
        s = self.ff1(s)
        if self.bits_others >= 16:
            s = self.relu(s)
        s = self.ff2(s)
        s = self.dropout(s)
        s = s + r

        return s, b, c


class QuantizedMultiHeadAttention(nn.Module):

    def __init__(self, H, D, dropout,
        weight_quant_dtype = 'quint8',
        weight_quant_method = 'per_tensor',
        activation_quant_dtype = 'quint8',
        activation_quant_method = 'per_tensor',
        dot_quant_dtype = 'quint8',
        dot_quant_method = 'per_channel',
        Av_quant_dtype = 'quint8',
        Av_quant_method = 'per_tensor',
        bits_Wq = 8,
        bits_Wk = 8,
        bits_Wv = 8,
        bits_dot = 8,
        bits_Av = 8,
        bits_Wo = 8
    ):
        super().__init__()

        self.H = H
        self.D = D
        self.Dh = D // H

        self.bits_dot = bits_dot
        self.bits_Av = bits_Av

        self.weight_quant_dtype = weight_quant_dtype
        self.weight_quant_method = weight_quant_method
        self.activation_quant_dtype = activation_quant_dtype
        self.activation_quant_method = activation_quant_method
        self.dot_quant_dtype = get_dtype_from_string(dot_quant_dtype)
        self.dot_quant_method = dot_quant_method
        self.Av_quant_dtype = get_dtype_from_string(Av_quant_dtype)
        self.Av_quant_method = Av_quant_method

        self.W_q = self.__create_linear_layer(bits_Wq)
        self.W_k = self.__create_linear_layer(bits_Wk)
        self.W_v = self.__create_linear_layer(bits_Wv)
        self.W_o = self.__create_linear_layer(bits_Wo)

        if bits_dot < 16:
            self.q_quantizer = FakeActivationQuantizer(
                bits_dot,
                self.dot_quant_dtype,
                self.dot_quant_method,
                channel_axis=3
            )
            self.k_quantizer = FakeActivationQuantizer(
                bits_dot,
                self.dot_quant_dtype,
                self.dot_quant_method,
                channel_axis=2
            )

        if bits_Av < 16:
            self.a_quantizer = FakeWeightQuantizer(
                bits_Av,
                self.Av_quant_dtype,
                self.Av_quant_method
            )
            self.v_quantizer = FakeActivationQuantizer(
                bits_Av,
                self.Av_quant_dtype,
                self.Av_quant_method
            )

        self.denom = math.sqrt(self.D)

        self.softmax = nn.Softmax(-1)
        self.dropout = nn.Dropout(dropout)

    def __create_linear_layer(self, bits):
        if bits < 16:
            return QuantizedLinear(self.D, self.D,
                bits=bits,
                weight_quant_dtype=self.weight_quant_dtype,
                weight_quant_method=self.weight_quant_method,
                activation_quant_dtype=self.activation_quant_dtype,
                activation_quant_method=self.activation_quant_method
            )
        else:
            return nn.Linear(self.D, self.D)

    def __call__(self, q, k, v, m=None):
        
        B = q.shape[0]
        D = self.D

        q = self.W_q(q)
        k = self.W_k(k)
        v = self.W_v(v)

        q = q.view(B, -1, self.H, self.Dh)
        k = k.view(B, -1, self.H, self.Dh)
        v = v.view(B, -1, self.H, self.Dh)

        q = torch.transpose(q, 1, 2)
        k = torch.transpose(k, 1, 2)
        v = torch.transpose(v, 1, 2)

        k = torch.transpose(k, -2, -1)

        if self.bits_dot < 16:
            q = self.q_quantizer(q)
            k = self.k_quantizer(k)

        a = torch.matmul(q, k)
        a = a / self.denom

        if m is not None:
            a = a.masked_fill(m, -float('inf'))

        a = self.softmax(a)
        a = self.dropout(a)

        if self.bits_Av < 16:
            a = self.a_quantizer(a)
            v = self.v_quantizer(v)

        o = torch.matmul(a, v)

        o = torch.transpose(o, 1, 2)
        o = o.reshape(B, -1, self.D)
        o = self.W_o(o)

        return o, a