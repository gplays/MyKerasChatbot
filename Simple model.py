#
#
#
from keras.layers import *

#
params = {}
# Input
src_text = Input(name=self.ids_inputs[0],
                 batch_shape=tuple([None, None]), dtype='int32')

# Embedding
src_embedding = Embedding(params['INPUT_VOCABULARY_SIZE'],
                          params['SOURCE_TEXT_EMBEDDING_SIZE'])(src_text)

# Encoder
lstm = [LSTM for i in range(params['N_LAYERS_ENCODER'] - 1)]
annotations = src_embedding
for n_layer in range(1, params['N_LAYERS_ENCODER']):
    annotations = Bidirectional(lstm[n_layer])(annotations)

# Decoder
next_words = Input()

state_below = Embedding(voc_size, hidden_size)(next_words)

ctx_mean = MaskedMean()(annotations)
annotations = MaskLayer()(annotations)

initial_state = Dense()(ctx_mean)
initial_memory = Dense()(ctx_mean)

input_attentional_decoder = [state_below, annotations, initial_state,
                             initial_memory]

sharedAttRNNCond = LstmAttCond

proj_h, x_att, alphas, h_state, h_memory = sharedAttRNNCond(
    input_attentional_decoder)

shared_Lambda_Permute = PermuteGeneral((1, 0, 2))
shared_proj_h_list = []
shared_reg_proj_h_list = []
h_states_list = [h_state]

for n_layer in range(1, params['N_LAYERS_DECODER']):
    current_rnn_input = [proj_h, shared_Lambda_Permute(x_att),
                         initial_state, initial_memory]

    shared_proj_h_list.append(LstmAttCond)
    current_rnn_output = shared_proj_h_list[-1](current_rnn_input)
    proj_h = current_rnn_output[0]
    h_states_list.append(current_rnn_output[1])
    h_memories_list.append(current_rnn_output[2])

shared_FC_mlp = TimeDistributed(Dense())
out_layer_mlp = shared_FC_mlp(proj_h)
shared_FC_ctx = TimeDistributed(Dense())
out_layer_ctx = shared_FC_ctx(x_att)
out_layer_ctx = shared_Lambda_Permute(out_layer_ctx)
shared_FC_emb = TimeDistributed(Dense())
out_layer_emb = shared_FC_emb(state_below)
additional_output = Add()([out_layer_mlp, out_layer_ctx, out_layer_emb])
shared_activation = Activation('tanh')

out_layer = shared_activation(additional_output)
shared_FC_soft = TimeDistributed(Dense())
softout = shared_FC_soft(out_layer)

self.model = Model(inputs=[src_text, next_words], outputs=softout)
