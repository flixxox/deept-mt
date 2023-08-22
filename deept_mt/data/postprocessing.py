from deept.data.postprocessing import register_postprocessing_fn

@register_postprocessing_fn('mt_postprocess')
def mt_postprocesser(name, value):

    from deept.util.globals import Context

    if 'src' in name:
        vocab = Context['vocab_src']
    elif (
        'tgt' in name or 
        'result' in name or
        'beam' in name or
        'out' in name):
        vocab = Context['vocab_tgt']
    else:
        raise ValueError(f'Unrecognized tensor name {name} for postprocessing!')

    value = value.numpy().tolist()

    processed = []

    for entry in value:
        entry = vocab.detokenize(entry)
        entry = vocab.remove_padding(entry)
        entry = ' '.join(entry)
        entry = entry.replace(vocab.EOS_TEXT, '')
        entry = entry.strip()
        processed.append(entry)

    return processed