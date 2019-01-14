# %%
# simple UnRollredRNN_Model

from mxnet import gluon, autograd
from mxnet.gluon import nn, rnn

from rnn.shared import *

# %% [markdown]
# ## Character RNN using gluon/lstm api
#
# Training sequence 2 sequence models using Gluon API


# %%
# Class to create model objects.
class GluonRNNModel(gluon.Block):
    """A model with an encoder, recurrent layer, and a decoder."""

    def __init__(self,
                 mode,
                 vocab_size,
                 num_embed,
                 num_hidden,
                 num_layers,
                 dropout=0.5,
                 **kwargs):
        super(GluonRNNModel, self).__init__(**kwargs)
        with self.name_scope():

            self.encoder = nn.Embedding(
                vocab_size, num_embed, weight_initializer=mx.init.Uniform(0.1))

            self.dropout = nn.Dropout(dropout)

            if mode == 'lstm':
                self.rnn = rnn.LSTM(
                    num_hidden,
                    num_layers,
                    dropout=dropout,
                    input_size=num_embed)
            elif mode == 'gru':
                self.rnn = rnn.GRU(
                    num_hidden,
                    num_layers,
                    dropout=dropout,
                    input_size=num_embed)
            else:
                self.rnn = rnn.RNN(
                    num_hidden,
                    num_layers,
                    activation='relu',
                    dropout=dropout,
                    input_size=num_embed)
            self.decoder = nn.Dense(vocab_size, in_units=num_hidden)
            self.num_hidden = num_hidden

    # define the forward pass of the neural network
    def forward(self, inputs, hidden):
        emb = self.dropout(self.encoder(inputs))
        output, hidden = self.rnn(emb, hidden)
        output = self.dropout(output)
        # print('output forward',output.shape)
        decoded = self.decoder(output.reshape((-1, self.num_hidden)))
        return decoded, hidden

    # Initial state of netork
    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)


# %%
# define the lstm
mode = 'lstm'
vocab_size = len(chars) + 1  # number of characters in vocab_size
embedsize = 500
hididen_units = 1000
number_layers = 2
clip = 0.2
epochs = 2  # use 200 epochs for good result
batch_size = 32
seq_length = 100  # sequence length
dropout = 0.4
log_interval = 64
rnn_save = 'checkpoints/gluonlstm_abc'  # checkpoints/gluonlstm_2 (prepared for seq_lenght 100, 200 epochs)

# %%
# GluonRNNModel
model = GluonRNNModel(mode, vocab_size, embedsize, hididen_units, number_layers,
                      dropout)
# initalise the weights of models to random weights
model.collect_params().initialize(mx.init.Xavier(), ctx=context)
# Adam trainer
trainer = gluon.Trainer(model.collect_params(), 'adam')
# softmax cros entropy loss
loss = gluon.loss.SoftmaxCrossEntropyLoss()

# %%


# prepares rnn batches
# The batch will be of shape is (num_example * batch_size) because of RNN uses sequences of input     x
# for example if we use (a1,a2,a3) as one input sequence , (b1,b2,b3) as another input sequence and (c1,c2,c3)
# if we have batch of 3, then at timestep '1'  we only have (a1,b1.c1) as input, at timestep '2' we have (a2,b2,c2) as input...
# hence the batchsize is of order
# In feedforward we use (batch_size, num_example)
def rnn_batch(data, batch_size):
    """Reshape data into (num_example, batch_size)"""
    nbatch = data.shape[0] // batch_size
    data = data[:nbatch * batch_size]
    data = data.reshape((batch_size, nbatch)).T
    return data


idx_nd = mx.nd.array(idx)
# convert the idex of characters
train_data_rnn_gluon = rnn_batch(idx_nd, batch_size).as_in_context(context)


# %%
# get the batch
def get_batch(source, i, seq):
    seq_len = min(seq, source.shape[0] - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len]
    return data, target.reshape((-1,))


# detach the hidden state, so we dont accidentally compute gradients
def detach(hidden):
    if isinstance(hidden, (tuple, list)):
        hidden = [i.detach() for i in hidden]
    else:
        hidden = hidden.detach()
    return hidden


# %%
def trainGluonRNN(epochs, train_data, seq=seq_length):
    for epoch in range(epochs):
        total_L = 0.0
        hidden = model.begin_state(
            func=mx.nd.zeros, batch_size=batch_size, ctx=context)
        for ibatch, i in enumerate(
                range(0, train_data.shape[0] - 1, seq_length)):
            data, target = get_batch(train_data, i, seq)
            hidden = detach(hidden)
            with autograd.record():
                output, hidden = model(data, hidden)
                L = loss(
                    output,
                    target)  # this is total loss associated with seq_length
                L.backward()

            grads = [i.grad(context) for i in model.collect_params().values()]
            # Here gradient is for the whole batch.
            # So we multiply max_norm by batch_size and seq_length to balance it.
            gluon.utils.clip_global_norm(grads, clip * seq_length * batch_size)

            trainer.step(batch_size)
            total_L += mx.nd.sum(L).asscalar()

            if ibatch % log_interval == 0 and ibatch > 0:
                cur_L = total_L / seq_length / batch_size / log_interval
                print('[Epoch %d Batch %d] loss %.2f', epoch + 1, ibatch, cur_L)
                total_L = 0.0
        model.save_parameters(rnn_save)


# %%
print('the train data shape is', train_data_rnn_gluon.shape)

# %%
# The train data shape
trainGluonRNN(epochs, train_data_rnn_gluon, seq=seq_length)

# %%
model.load_parameters(rnn_save, context)


# %%
# evaluates a seqtoseq model over input string
def evaluate_seq2seq(model, input_string, seq_length, batch_size):
    idx = [char_indices[c] for c in input_string]
    if (len(input_string) != seq_length):
        raise ValueError("input string should be equal to sequence length")
    hidden = model.begin_state(
        func=mx.nd.zeros, batch_size=batch_size, ctx=context)
    sample_input = mx.nd.array(np.array([idx[0:seq_length]]).T, ctx=context)
    output, hidden = model(sample_input, hidden)
    index = mx.nd.argmax(output, axis=1)
    index = index.asnumpy()
    return [indices_char[char] for char in index]


# %%
# maps the input sequence to output sequence
def mapInput(input_str, output_str):
    for i, _ in enumerate(input_str):
        partial_input = input_str[:i + 1]
        partial_output = output_str[i:i + 1]
        print(partial_input + "->" + partial_output[0])


# %%
test_input = 'probably the time is at hand when it will be once and again understood WHAT has actually sufficed an'
print(len(test_input))
result = evaluate_seq2seq(model, test_input, seq_length, 1)
mapInput(test_input, result)


# %%
# a nietzsche like text generator
def generate_random_text(model, input_string, seq_length, batch_size,
                         sentence_length):
    count = 0
    new_string = ''
    cp_input_string = input_string
    hidden = model.begin_state(
        func=mx.nd.zeros, batch_size=batch_size, ctx=context)
    while count < sentence_length:
        idx = [char_indices[c] for c in input_string]
        if (len(input_string) != seq_length):
            print(len(input_string))
            raise ValueError('there was a error in the input ')
        sample_input = mx.nd.array(np.array([idx[0:seq_length]]).T, ctx=context)
        output, hidden = model(sample_input, hidden)
        index = mx.nd.argmax(output, axis=1)
        index = index.asnumpy()
        count = count + 1
        new_string = new_string + indices_char[index[-1]]
        input_string = input_string[1:] + indices_char[index[-1]]
    print(cp_input_string + new_string)


# %%
generate_random_text(
    model,
    "probably the time is at hand when it will be once and again understood WHAT has actually sufficed an",
    seq_length, 1, 200)
