
def build_dataset(params):

    dataset = None
    params['input_vocabulary_size'] = dataset.vocabulary_len[params[
        'inputs_ids_dataset'][0]]

