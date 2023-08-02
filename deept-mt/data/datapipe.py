
import torch

from torchdata.datapipes.iter import IterDataPipe

from deept.util.debug import my_print
from deept.util.globals import Context
from deept.data.datapipe import register_len_fn
from deept.data.datapipe import register_dp_collate
from deept.data.datapipe import register_dp_decoding
from deept.data.datapipe import register_dp_preprocessing


@register_len_fn('mt_tgt_len_fn')
def mt_length_function(item):
    return len(item['tgt'])+1


@register_dp_decoding('text')
class WebDatasetTextDecoderIterDataPipe(IterDataPipe):

    def __init__(self, source_dp: IterDataPipe):
        super().__init__()
        self.source_dp = source_dp

    @staticmethod
    def create_from_config(config, source_dp):
        return WebDatasetTextDecoderIterDataPipe(source_dp)

    def __iter__(self):
        for d in self.source_dp:
            yield self.webdataset_text_decode(d)

    def webdataset_text_decode(self, item):
        key, value = item
        return key, value.read().decode('utf-8')

    def __len__(self):
        raise NotImplementedError('Error! Do not invoke len(datapipe).')


class MTVocabulary:

    UNK = "<UNK>"
    PAD = "<PAD>"
    EOS = "</S>"

    special_tokens = [
        UNK,
        PAD,
        EOS
    ]

    def __init__(self, vocab, vocab_size, vocab_rev, vocab_path):
        
        self.vocab = vocab
        self.vocab_size = vocab_size
        self.vocab_rev = vocab_rev
        self.vocab_path = vocab_path

        self.PAD = self.vocab[MTVocabulary.PAD]
        self.UNK = self.vocab[MTVocabulary.UNK]
        self.EOS = self.vocab[MTVocabulary.EOS]

        self.UNK_TEXT = MTVocabulary.UNK
        self.PAD_TEXT = MTVocabulary.PAD
        self.EOS_TEXT = MTVocabulary.EOS

    @staticmethod
    def create_vocab(vocab_path):

        vocab = MTVocabulary.read_from_pickle(vocab_path)
        vocab = MTVocabulary.append_special_tokens(vocab)
        vocab = MTVocabulary.remove_sos_symbol_from_vocabs(vocab)

        vocab_size = len(vocab.items())
        vocab_rev = {y:x for x,y in vocab.items()}

        return MTVocabulary(vocab, vocab_size, vocab_rev, vocab_path)

    @staticmethod
    def read_from_pickle(vocab_path):
        import pickle
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        return vocab

    @staticmethod
    def append_special_tokens(vocab):

        count = 0
        for special_token in MTVocabulary.special_tokens:
            if special_token not in vocab.keys():
                count += 1

        MTVocabulary.increment_dictionary(vocab, count)

        new_index = 0
        for special_token in MTVocabulary.special_tokens:
            if special_token not in vocab.keys():
                my_print(f'Inserting special token {special_token} to vocabulary at position {new_index}.')
                vocab[special_token] = new_index
                new_index += 1

        vocab = {k: v for k, v in sorted(vocab.items(), key=lambda item: item[1])}

        return vocab

    @staticmethod
    def increment_dictionary(dictionary, increment):

        assert isinstance(dictionary, dict)
        assert isinstance(increment, int)

        for entry in dictionary:

            assert isinstance(dictionary[entry], int)

            dictionary[entry] += increment
    
    @staticmethod
    def remove_sos_symbol_from_vocabs(vocab):
        if "<S>" in vocab:
            del vocab["<S>"]
        return vocab

    def print_vocab(self):
        for k,v in self.vocab.items():
            my_print(k,v)

    def tokenize(self, inp):
        if isinstance(inp, list):
            return self.tokenize_list(inp)
        elif isinstance(inp, str):
            return self.tokenize_word(inp)
        else:
            assert True == False, 'Got unknown input for tokenization.'

    def tokenize_list(self, line):

        assert isinstance(line, list)

        tokenized = []
        for word in line:
            tokenized.append(self.tokenize_word(word))

        return tokenized

    def tokenize_word(self, word):
        if word in self.vocab:
            return int(self.vocab[word])
        else:
            return int(self.vocab[MTVocabulary.UNK])


    def detokenize(self, inp):
        if isinstance(inp, list):
            return self.detokenize_list(inp)
        elif isinstance(inp, str):
            return self.detokenize_word(inp)
        else:
            assert True == False, 'Got unknown input for detokenization.'

    def detokenize_list(self, line):

        assert isinstance(line, list)

        detokenized = []
        for word in line:
            detokenized.append(self.detokenize_word(word))

        return detokenized

    def detokenize_word(self, word):
        return self.vocab_rev[word]

    
    def remove_padding(self, sentence):
        ret = []
        for word in sentence:
            if word != MTVocabulary.PAD:
                ret.append(word)
        return ret


@register_dp_preprocessing('mt_preprocess')
class MTPreprocesserIterDataPipe(IterDataPipe):

    def __init__(self, 
        source_dp,
        vocab_src,
        vocab_tgt
    ):
        super().__init__()

        self.source_dp = source_dp
        self.vocab_src = vocab_src
        self.vocab_tgt = vocab_tgt

    @staticmethod
    def create_from_config(config, source_dp):

        from torchdata.datapipes.iter import Filter

        if not Context.has_context('vocab_src'):
            vocab_src = MTVocabulary.create_vocab(config['vocab_src'])
            Context.add_context('vocab_src', vocab_src)
            my_print(f'Vocab size source {vocab_src.vocab_size}!')
        else:
            vocab_src = Context['vocab_src']

        if not Context.has_context('vocab_tgt'):
            vocab_tgt = MTVocabulary.create_vocab(config['vocab_tgt'])
            Context.add_context('vocab_tgt', vocab_tgt)
            my_print(f'Vocab size target {vocab_tgt.vocab_size}!')
        else:
            vocab_tgt = Context['vocab_tgt']

        if not Context.has_context('pad_index'):
            Context.add_context('pad_index', vocab_src.PAD)

        assert vocab_src.PAD == vocab_tgt.PAD

        pipe = MTPreprocesserIterDataPipe(
            source_dp,
            vocab_src,
            vocab_tgt,
        )

        pipe = MTLengthFilterIterDataPipe.create_from_config(config, pipe)

        return pipe

    def __iter__(self):
        for item in self.source_dp:
            yield self.mt_preprocess(
                self.normalize_keys(item)
            )

    def normalize_keys(self, item):
        
        assert isinstance(item, dict), """The webdataset format presets that every sample
            is a dictionary."""

        assert '__key__' in item.keys(), """The webdataset format presets that every sample
            is a dictionary and contains a __key__ entry."""

        idx = item['__key__'].split('/')[-1]

        assert 'sample' in idx, """At the moment we expect you to name each sample of the webdataset sampleXXXXX."""

        idx = int(idx.split('/')[-1].split('sample')[-1])

        normalized_dict = {
            '__key__': idx
        }

        for k, v in item.items():
            if 'source' in k:
                normalized_dict['src'] = v
            elif 'target' in k:
                normalized_dict['tgt'] = v

        return normalized_dict

    def mt_preprocess(self, item):

        item['src'] = item['src'].strip().replace("\n", "").split(" ")
        item['tgt'] = item['tgt'].strip().replace("\n", "").split(" ")

        item['src'] = self.vocab_src.tokenize(item['src'])
        item['tgt'] = self.vocab_src.tokenize(item['tgt'])

        return item
    
    def __len__(self):
        raise NotImplementedError('Error! Do not invoke len(datapipe).')


class MTLengthFilterIterDataPipe(IterDataPipe):

    def __init__(self, 
        source_dp,
        min_length,
        max_length
    ):
        super().__init__()

        self.source_dp = source_dp

        self.min_length = min_length
        self.max_length = max_length

    @staticmethod
    def create_from_config(config, source_dp):

        min_length = config['min_sample_size', 0]
        max_length = config['max_sample_size', None]

        return MTLengthFilterIterDataPipe(
            source_dp,
            min_length,
            max_length
        )

    def __iter__(self):
        for item in self.source_dp:
            if self.length_filter(item):
                yield item

    def length_filter(self, item):
        src_len = len(item['src'])+2
        tgt_len = len(item['tgt'])+1
        return (
            src_len >= self.min_length and
            src_len <= self.max_length and
            tgt_len >= self.min_length and
            tgt_len <= self.max_length
        )
    
    def __len__(self):
        raise NotImplementedError('Error! Do not invoke len(datapipe).')


@register_dp_collate('mt_collate')
class MTCollaterIterDataPipe(IterDataPipe):

    def __init__(self, source_dp):
        super().__init__()
        self.source_dp = source_dp
        self.vocab_src = Context['vocab_src']
        self.vocab_tgt = Context['vocab_tgt']

    @staticmethod
    def create_from_config(config, source_dp):
        return MTCollaterIterDataPipe(source_dp)

    def __iter__(self):
        for item in self.source_dp:
            yield self.mt_collate(item)

    def mt_collate(self, item):

        dict_collated = {
            '__keys__': [x['__key__'] for x in item]
        }

        pad_index = self.vocab_src.PAD
        eos_index_src = self.vocab_src.EOS
        eos_index_tgt = self.vocab_tgt.EOS

        max_s = max([len(x['src']) for x in item])
        max_t = max([len(x['tgt']) for x in item])

        s = [[eos_index_src] + x['src'] + [eos_index_src] + [pad_index] * (max_s - len(x['src'])) for x in item]
        t = [[eos_index_tgt] + x['tgt'] + [pad_index] * (max_t - len(x['tgt'])) for x in item]
        o = [x['tgt'] + [eos_index_tgt] + [pad_index] * (max_t - len(x['tgt'])) for x in item]

        dict_collated['src'] = torch.tensor(s)
        dict_collated['tgt'] = torch.tensor(t)
        dict_collated['out'] = torch.tensor(o)

        return dict_collated

    def __len__(self):
        raise NotImplementedError('Error! Do not invoke len(datapipe).')