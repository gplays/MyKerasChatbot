import json
import logging

from keras.layers import *
from keras.models import model_from_json, Model
from keras.regularizers import l2


# from keras_wrapper.cnn_model import CNN_Model
# from keras_wrapper.cnn_model import Model_Wrapper


class ChatBotModel(object):
    """
    ChatBotModel object constructor.

    :param params: all hyperparameters of the model.
    :param vocabularies: vocabularies used for GLOVE word embedding


    """

    def __init__(self, cfg, vocabularies=None,
                 clear_dirs=True):
        self.params = cfg["general"]
        self._process_params()
        # Prepare directory for storing results, structure and weights
        self._prepare_storing()

        self.model_params = ModelParameterHandler(cfg["model_params"])

        self._load_or_create_model()

        self._maybe_load_weights_from_file()

        self.print_self()

        self._setOptimizer()

    def _process_params(self):
        params = self.params

        self.verbose = params["verbose"]

        self.data_dir = params["data_dir"]
        self.out_dir = params["out_dir"]
        self.vocabulary = params["vocabulary"]
        self.pretrained_vectors = params["pretrained_vectors"]
        self.inputs = params["inputs"]
        self.store_model = params["store_model"]
        self.prebuilt_model = params["prebuilt_model"]
        self.pretrained_weights = params["pretrained_weights"]

        self.optimizer = params.get("optimizer", "Adam")
        self.lr = params.get("lr", 0.001)
        self.clipnorm = params.get("clipnorm", 1)
        self.clipvalue = params.get("clipvalue", 0)
        self.momentum = params.get("momentum", 0)
        self.nesterov = params.get("nesterov", False)
        self.rho = params.get("rho", 0.9)
        self.beta_1 = params.get("beta_1", 0.9)
        self.beta_2 = params.get("beta_2", 0.999)
        self.decay = params.get("decay", None)
        self.gamma = params.get("gamma", 0.8)

        self.model_name = params.get("name", "")
        self.vocabulary_size = params["vocabulary_size"]
        self.speakers_size = params["speakers_size"]
        self.text_embedding_hidden_size = params["text_embedding_hidden_size"]
        self.speaker_embedding_hidden_size = params[
            "speaker_embedding_hidden_size"]
        self.LSTM = params["LSTM"]
        self.n_layers_decoder = params["n_layers_decoder"]

    def _setOptimizer(self, **kwargs):
        # TODO
        pass

    def _prepare_storing(self):
        self.model_name = self.compute_model_name()
        store_path = self.store_model

    def _load_or_create_model(self):
        structure_path = self.prebuilt_model
        if structure_path:
            # Load a .json model
            if self.verbose > 0:
                logging.info("<<< Loading model structure from file " +
                             structure_path + " >>>")
            self.model = model_from_json(open(structure_path).read())
        else:
            self._build_model()

    def _maybe_load_weights_from_file(self):
        weights_path = self.pretrained_weights
        if weights_path:
            if self.verbose > 0:
                logging.info(
                    "<<< Loading weights from file " + weights_path + " >>>")
            self.model.load_weights(weights_path)

    def print_self(self):
        """
        Print summary of the model
        :return:
        :rtype:
        """
        if self.verbose > 0:
            print(str(self))
            self.model.summary()

    def compute_model_name(self):
        """
        Generate a name for the model from parameters
        :return: name of the model
        :rtype: str
        """
        if not self.model_name:
            # TODO compile name from parameters + timestamp?
            self.model_name = "my_model"

    def _build_model(self):
        self.layers = {}
        layers = self.layers
        self.decoder_isLSTM = self.LSTM

        # Store inputs and outputs names
        self.ids_inputs = ["src_text", "src_speaker", "tgt_text", "tgt_speaker"]
        self.ids_outputs = ["target_text"]
        src_text = Input(name=self.ids_inputs[0],
                         batch_shape=tuple([None, None]),
                         dtype='int32')
        src_speaker = Input(name="src_speaker",
                            batch_shape=tuple([None, None]),
                            dtype='int32')
        tgt_text = Input(name=self.ids_inputs[2],
                         batch_shape=tuple([None, None]),
                         dtype='int32')
        tgt_speaker = Input(name=self.ids_inputs[3],
                            batch_shape=tuple([None, None]),
                            dtype='int32')

        input_encoder = [src_text, src_speaker]
        input_decoder = [tgt_text, tgt_speaker]
        inputs = input_encoder + input_decoder

        # Encoding
        self.model_params.set_step("encoder")
        self._init_word_embedding()
        self._build_speaker_embedding()
        src_sentence_embedding = layers["word_embedding"](src_text)
        src_speaker_embedding = layers["speaker_embedding"](src_speaker)
        output_encoder = self._build_encoder(src_sentence_embedding,
                                             src_speaker_embedding)

        self.model_params.set_step("decoder")
        output_decoder = self._build_decoder(output_encoder)
        output = self._output(output_decoder)

        self.model_params.set_step("infer")
        self._sampling(inputs, output_encoder, output)

        self.model = Model(input=inputs, output=output)

        self.setOptimizer()

    def _init_word_embedding(self):
        word_vectors = self.maybe_load_pretrained_word_vectors()

        rand = np.random.rand
        with open(self.vocabulary) as f:
            voc = json.load(f)
        voc_size = len(voc)

        hidden_size = self.params["model_struct"]['text_embedding_hidden_size']
        embedding_weights = rand(voc_size, hidden_size)

        if word_vectors is not None:
            for word, index in voc.items():
                if word_vectors.get(word) is not None:
                    embedding_weights[index, :] = word_vectors[word]

        self.layers["embedding_layer"] = Embedding(
            voc_size, hidden_size,
            name='word_embedding',
            weights=[embedding_weights],
            **self.get_cfg("word_embedding"))

    def _build_speaker_embedding(self):
        self.layers["speaker_embedding"] = Embedding(
            self.speakers_size,
            self.speaker_embedding_hidden_size,
            name='speaker_embedding',
            **self.get_cfg("speaker_embedding"))

    def _build_encoder(self, *args):
        encoder = self._build_lstm_encoder(*args)
        if self.params['cnn_encoder']:
            encoder_cnn = self._build_cnn_encoder(*args)

            encoder = Concatenate([encoder, encoder_cnn])
        self.preprocessed_size = encoder.output_shape[-1]
        return encoder

    def _build_lstm_encoder(self, sentence_embedding):
        params = self.params

        if params['attention']:
            lstm = AttLSTM
        else:
            lstm = LSTM

        lstm = lstm(units=params['lstm_encoder_hidden_size'],
                    return_sequences=True,
                    **self.get_cfg("encoder_lstm"))

        out_layer = Bidirectional(lstm,
                                  name='bidirectional_encoder',
                                  merge_mode='concat')(sentence_embedding)

        return out_layer

    def _build_cnn_encoder(self, sentence_embedding):
        convolutions = []
        params = self.params
        for filter_len in params['filter_sizes']:
            conv_layer = Convolution1D(filters=params['num_filters'],
                                       kernel_size=filter_len,
                                       activation=params['cnn_activation'],
                                       **self.get_cfg("encoder_cnn"))
            conv = conv_layer(sentence_embedding)
            pool = MaxPooling1D()(conv)
            # pool = Regularize(pool, params, name='pool_' + str(filter_len))
            convolutions.append(Flatten()(pool))
        if len(convolutions) > 1:
            out_layer = merge(convolutions, mode='concat')
        else:
            out_layer = convolutions[0]

        return out_layer

    def _build_decoder(self, inputs, outputEncoder):
        # 3. Decoder
        # 3.1.1. Previously generated words as inputs for training -> Teacher
        #  forcing


        # 3.1.2. Target word embedding
        tgt_sentence_embedding = self.layers["word_embedding"](inputs[0])
        tgt_speaker_embedding = self.layers["speaker_embedding"](inputs[1])

        self.state_below = tgt_sentence_embedding
        self.tgt_speaker_embedding = tgt_speaker_embedding

        inputDecoder = [tgt_sentence_embedding, tgt_speaker_embedding]

        self._decoder_initialization(outputEncoder,
                                     inputDecoder)

    def _decoder_initialization(self, outputEncoder, inputDecoder):
        # 3.2. Decoder's RNN initialization perceptrons with ctx mean
        params = self.params
        annotations = outputEncoder
        state_below = inputDecoder[0]
        tgt_speaker_embedding = inputDecoder[1]
        # TODO check correctness
        annotations = Concat(annotations, tgt_speaker_embedding)

        decoder_hidden_size = params['decoder_hidden_size']

        ctx_mean = MaskedMean()(annotations)

        # We may want the padded annotations
        annotations = MaskLayer()(annotations)

        init_layers = self.model_params.get("encoder_mean")
        init_layers = init_layers.get("init_layers", [])

        if len(init_layers) > 0:

            for n_layer_init in range(len(init_layers) - 1):

                ctx_mean_layer = Dense(decoder_hidden_size,
                                       name='init_layer_%d' % n_layer_init,
                                       activation=init_layers[n_layer_init],
                                       **self.get_cfg("encoder_mean"))
                ctx_mean = ctx_mean_layer(ctx_mean)

            initial_state_layer = Dense(decoder_hidden_size,
                                        name='initial_state',
                                        activation=init_layers[-1],
                                        **self.get_cfg("decoder_initial_state"))
            initial_state = initial_state_layer(ctx_mean)

            initial_memory_layer = Dense(decoder_hidden_size,
                                         name='initial_memory',
                                         activation=init_layers[-1],
                                         **self.get_cfg(
                                             "decoder_initial_memory"))
            initial_memory = initial_memory_layer(ctx_mean)

            input_attentional_decoder = [state_below, annotations,
                                         initial_state, initial_memory]
        else:
            # Initialize to zeros vector
            input_attentional_decoder = [state_below, annotations]
            initial_state = ZeroesLayer(decoder_hidden_size)(ctx_mean)
            input_attentional_decoder.append(initial_state)
            input_attentional_decoder.append(initial_state)

        return input_attentional_decoder

    def _att_decoder(self, input_attentional_decoder):
        params = self.params
        self.rnn_decoder_cell = AttLSTMCond

        ## 3.3. Attentional decoder

        self.layers['AttRNNCond'] = self.rnn_decoder_cell(
            params['decoder_hidden_size'],
            return_sequences=True,
            return_extra_variables=True,
            return_states=True,
            num_inputs=len(input_attentional_decoder),
            name='decoder_Att' + params['decoder_rnn_type'] + 'Cond',
            **self.get_cfg("decoder_rnn_cond"), )

        rnn_output = self.layers['AttRNNCond'](input_attentional_decoder)

        self.layers['Lambda_Permute'] = PermuteGeneral((1, 0, 2))

        return rnn_output

    def _deep_decoder(self, rnn_output, input_attentional_decoder):
        # 3.4. Possibly deep decoder
        initial_state = input_attentional_decoder[2]
        params = self.params

        proj_h = rnn_output[0]
        x_att = rnn_output[1]
        h_state = rnn_output[3]
        h_memory = rnn_output[4]
        initial_memory = input_attentional_decoder[3]

        shared_proj_h_list = []

        h_states_list = [h_state]
        h_memories_list = [h_memory]

        # TODO understand the permutation
        for n_layer in range(1, params['n_layers_decoder']):
            current_rnn_input = [proj_h, self.layers['Lambda_Permute'](x_att),
                                 initial_state, initial_memory]

            name = "decoder_{}_Cond_{}".format(params['decoder_rnn_type'],
                                               n_layer)

            shared_proj_h = self.rnn_decoder_cell(
                params['decoder_hidden_size'],
                return_sequences=True,
                return_states=True,
                num_inputs=len(current_rnn_input),
                name=name,
                **self.get_cfg("deep_decoder"), )

            current_rnn_output = shared_proj_h(current_rnn_input)
            current_proj_h = current_rnn_output[0]
            h_states_list.append(current_rnn_output[1])

            shared_proj_h_list.append(shared_proj_h)

            h_memories_list.append(current_rnn_output[2])

            proj_h = Add()([proj_h, current_proj_h])

            self.h_states_list = h_states_list
            self.h_memories_list = h_memories_list
            self.proj_h = proj_h
            self.shared_proj_h_list = shared_proj_h_list

    def _skip_connections_encoder_output(self, proj_h, x_att, state_below):
        # 3.5. Skip connections between encoder and output layer

        params = self.params
        trainable = self.params['trainable_decoder']
        hidden_size = self.params['text_embedding_size']

        ######
        fc_mlp = Dense(hidden_size,
                       activation='linear',
                       **self.get_cfg("logit_lstm"), )
        self.layers['FC_mlp'] = TimeDistributed(fc_mlp,
                                                trainable=trainable,
                                                name='logit_lstm')
        out_layer_mlp = self.layers['FC_mlp'](proj_h)

        ######
        fc_ctx = Dense(hidden_size,
                       activation='linear',
                       **self.get_cfg("logit_ctx"), )
        self.layers['FC_ctx'] = TimeDistributed(fc_ctx,
                                                trainable=trainable,
                                                name='logit_ctx')
        out_layer_ctx = self.layers['FC_ctx'](x_att)
        out_layer_ctx = self.layers['Lambda_Permute'](out_layer_ctx)

        ######
        fc_emb = Dense(hidden_size,
                       activation='linear',
                       **self.get_cfg("logit_emb"), )
        self.layers['FC_emb'] = TimeDistributed(fc_emb,
                                                trainable=trainable,
                                                name='logit_emb')
        out_layer_emb = self.layers['FC_emb'](state_below)
        ######

        self.layers['additional_output_merge'] = Add(name='additional_output')

        additional_output = self.layers['additional_output_merge'](
            [out_layer_mlp, out_layer_ctx, out_layer_emb])

        self.layers['activation'] = Activation('tanh')

        out_layer = self.layers['activation'](additional_output)

        return out_layer

    def _deep_output(self, out_layer):
        # 3.6 Optional deep output layer
        params = self.params
        trainable = self.params['trainable_decoder']

        self.shared_deep_list = []

        deep_out_cfg = self.get_cfg("deep_output").get("layers_cfg", [])

        for i, (activation, dimension) in enumerate(
                params['deep_output_layers']):
            name = 'decoder_output_{}_{}'.format(activation, i)
            deep_output_layer = Dense(dimension,
                                      activation=activation,
                                      **self.get_cfg("deep_output"))
            deep_output_layer = TimeDistributed(deep_output_layer,
                                                trainable=trainable,
                                                name=name)

            self.layers['deep_list'].append(deep_output_layer)
            out_layer = deep_output_layer(out_layer)

        return out_layer

    def _output(self, out_layer):
        # 3.7. Output layer: Softmax
        params = self.params
        trainable = self.params['trainable_decoder']

        fc_soft = Dense(params['output_vocabulary_size'],
                        activation=self.get_cfg("softout")["activation"],
                        name=self.get_cfg("softout")["activation"])
        self.layers['FC_soft'] = TimeDistributed(fc_soft,
                                                 trainable=trainable,
                                                 name=self.ids_outputs[0])
        softout = self.layers['FC_soft'](out_layer)

        return softout

    def _sampling(self, input, output_encoder, output):
        ##################################################################
        #                         SAMPLING MODEL                         #
        ##################################################################
        """Now that we have the basic training model ready, let's prepare the
        model for applying decoding
        The beam-search model will include all the minimum required set of
        layers (decoder stage) which offer the possibility to generate the next
        state in the sequence given a pre-processed input (encoder stage)
        First, we need a model that outputs the preprocessed input + initial h
        state for applying the initial forward pass"""

        params = self.params
        softout = output
        annotations = output_encoder
        annotations = self._speaker_merge(annotations,
                                          self.tgt_speaker_embedding)
        h_states_list = self.h_states_list
        h_memories_list = self.h_memories_list
        state_below = self.state_below

        # TODO is tgt_text needed? Is it only start word?
        model_init_input = input
        tgt_text = input[2]
        model_init_output = [softout, annotations] + self.h_states_list
        model_init_output += self.h_memories_list

        # No alpha used
        # if self.return_alphas:
        #     model_init_output.append(alphas)

        self.model_init = Model(inputs=model_init_input,
                                outputs=model_init_output)

        # Store inputs and outputs names for model_init
        self.ids_inputs_init = self.ids_inputs
        ids_states_names = ['next_state_' + str(i) for i in
                            range(len(h_states_list))]

        # first output must be the output probs.
        self.ids_outputs_init = self.ids_outputs + [
            'preprocessed_input'] + ids_states_names

        ids_memories_names = ['next_memory_' + str(i) for i in
                              range(len(h_memories_list))]
        self.ids_outputs_init += ids_memories_names

        # Second, we need to build an additional model with the capability to
        #  have the following inputs:
        #   - preprocessed_input
        #   - prev_word
        #   - prev_state
        # and the following outputs:
        #   - softmax probabilities
        #   - next_state

        # TODO check size after concat with tgt_speaker
        preprocessed_size = self.preprocessed_size

        # Define inputs
        n_deep_decoder_layer_idx = 0
        preprocessed_annotations = Input(name='preprocessed_input',
                                         shape=tuple([None, preprocessed_size]))
        prev_h_states_list = [Input(name='prev_state_' + str(i),
                                    shape=tuple(
                                        [params['decoder_hidden_size']]))
                              for i in range(len(h_states_list))]

        input_attentional_decoder = [state_below, preprocessed_annotations,
                                     prev_h_states_list[
                                         n_deep_decoder_layer_idx]]

        prev_h_memories_list = [Input(name='prev_memory_' + str(i),
                                      shape=tuple(
                                          [params['decoder_hidden_size']]))
                                for i in range(len(h_memories_list))]

        input_attentional_decoder.append(
            prev_h_memories_list[n_deep_decoder_layer_idx])

        # Apply decoder
        rnn_output = self.layers['AttRNNCond'](input_attentional_decoder)
        proj_h = rnn_output[0]
        x_att = rnn_output[1]
        h_states_list = [rnn_output[3]]
        h_memories_list = [rnn_output[4]]

        for rnn_decoder_layer in self.shared_proj_h_list:
            # TODO Verify index coherence. Maybe implement in a safer way
            n_deep_decoder_layer_idx += 1
            input_rnn_decoder_layer = [proj_h,
                                       self.layers['Lambda_Permute'](x_att),
                                       prev_h_states_list[
                                           n_deep_decoder_layer_idx],
                                       prev_h_memories_list[
                                           n_deep_decoder_layer_idx]]

            current_rnn_output = rnn_decoder_layer(input_rnn_decoder_layer)
            current_proj_h = current_rnn_output[0]
            h_states_list.append(current_rnn_output[1])  # h_state

            h_memories_list.append(current_rnn_output[2])  # h_memory

            proj_h = Add()([proj_h, current_proj_h])

        out_layer_mlp = self.layers['FC_mlp'](proj_h)
        out_layer_ctx = self.layers['FC_ctx'](x_att)
        out_layer_ctx = self.layers['Lambda_Permute'](out_layer_ctx)
        out_layer_emb = self.layers['FC_emb'](state_below)

        additional_output = self.layers['additional_output_merge'](
            [out_layer_mlp, out_layer_ctx, out_layer_emb])
        out_layer = self.layers['activation'](additional_output)

        for deep_out_layer in self.layers['deep_list']:
            out_layer = deep_out_layer(out_layer)

        # Softmax
        softout = self.layers['FC_soft'](out_layer)
        model_next_inputs = [tgt_text,
                             preprocessed_annotations] + prev_h_states_list
        model_next_outputs = [softout, preprocessed_annotations] + h_states_list
        model_next_inputs += prev_h_memories_list
        model_next_outputs += h_memories_list

        # if self.return_alphas:
        #     model_next_outputs.append(alphas)

        self.model_next = Model(inputs=model_next_inputs,
                                outputs=model_next_outputs)

        # TODO Understand next part usefullness: maybe only for keras Wrapper
        # Store inputs and outputs names for model_next
        # first input must be previous word
        # TODO check indexing, dangerous here "next_words"
        self.ids_inputs_next = [self.ids_inputs[2]] + ['preprocessed_input']
        # first output must be the output probs.
        self.ids_outputs_next = self.ids_outputs + ['preprocessed_input']
        # Input -> Output matchings from model_init to model_next and from
        # model_next to model_next
        self.matchings_init_to_next = {
            'preprocessed_input': 'preprocessed_input'}
        self.matchings_next_to_next = {
            'preprocessed_input': 'preprocessed_input'}
        # append all next states and matchings

        for n_state in range(len(prev_h_states_list)):
            self.ids_inputs_next.append('prev_state_' + str(n_state))
            self.ids_outputs_next.append('next_state_' + str(n_state))
            self.matchings_init_to_next[
                'next_state_' + str(n_state)] = 'prev_state_' + str(n_state)
            self.matchings_next_to_next[
                'next_state_' + str(n_state)] = 'prev_state_' + str(n_state)

        for n_memory in range(len(prev_h_memories_list)):
            self.ids_inputs_next.append('prev_memory_' + str(n_memory))
            self.ids_outputs_next.append('next_memory_' + str(n_memory))
            self.matchings_init_to_next[
                'next_memory_' + str(n_memory)] = 'prev_memory_' + str(
                n_memory)
            self.matchings_next_to_next[
                'next_memory_' + str(n_memory)] = 'prev_memory_' + str(
                n_memory)

    def _speaker_merge(self, annotations, speaker_embedding):
        return Concat(annotations, speaker_embedding)

    def maybe_load_pretrained_word_vectors(self):
        pretrained_vector_file = self.pretrained_vectors
        expected_size = self.text_embedding_hidden_size
        if pretrained_vector_file:
            if self.verbose > 0:
                logging.info("<<< Loading pretrained word vectors from file " +
                             pretrained_vector_file + " >>>")
            with open(pretrained_vector_file, 'r') as f:
                word_vectors = {}
                lines = iter(f)
                voc_size, embedding_size = [int(val) for val in
                                            next(lines).split()]

                for line in f:
                    splitLine = line.split()
                    word = splitLine[0]
                    embedding = np.array([float(val) for val in splitLine[1:]])
                    word_vectors[word] = embedding

            try:
                assert (self.text_embedding_hidden_size == embedding_size)
            except AssertionError:
                self.text_embedding_hidden_size = embedding_size
                logging.warning(
                    "text_embedding_hidden_size reset to {} to match "
                    "the embedding size of pretrained vectors".format(
                        embedding_size))

        else:
            word_vectors = None
        return word_vectors

    def get_cfg(self, layer_name):
        """
        Shortcut to query the ParameterHandler for the configuration of a layer
        :param layer_name: Name of the layer
        :type layer_name: str
        :return: dict of parameters
        :rtype: dict
        """
        return self.model_params.get(layer_name)


class ModelParameterHandler(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self._replace_aliases()
        self._setup_regularizers()
        self.set_step("decoder")

    def _replace_aliases(self):
        aliases = self.cfg["aliases"]
        for k, v in self.cfg.items():
            for k2, v2 in v.items():
                try:
                    if v2 in aliases:
                        self.cfg[k][k2] = aliases.get(v2, v2)
                except TypeError:
                    pass

    def _setup_regularizers(self):
        # change string regularization parameters to usable ones
        for k, v in self.cfg.items():
            for k2, v2 in v.items():
                if k2[-11:] == "regularizer":
                    self.cfg[k][k2] = l2(v2)

    def set_step(self, step_string):
        self.step = step_string

    def get(self, layer_name):
        parameters = {}
        main_dict = self.cfg.get("main", {})
        step_dict = self.cfg.get(self.step, {})
        layer_dict = self.cfg.get(layer_name, {})
        for k, v in main_dict.items():
            parameters[k] = v
        for k, v in step_dict.items():
            parameters[k] = v
        for k, v in layer_dict.items():
            parameters[k] = v
        return parameters


if __name__ == "__main__":
    print("Running Test")
    with open("config/config.json") as f:
        cfg = json.load(f)
    cb = ChatBotModel(cfg)
