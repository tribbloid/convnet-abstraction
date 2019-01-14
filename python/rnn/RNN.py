#%%
#simple UnRollredRNN_Model

import os
from mxnet import gluon, autograd
from mxnet.gluon import Block, nn
from mxnet import ndarray as F

from rnn.shared import *


class UnRolledRNN_Model(Block):

    def __init__(self, vocab_size, num_embed, num_hidden, **kwargs):
        super(UnRolledRNN_Model, self).__init__(**kwargs)
        self.num_embed = num_embed
        self.vocab_size = vocab_size

        # use name_scope to give child Blocks appropriate names.
        # It also allows sharing Parameters between Blocks recursively.
        with self.name_scope():
            self.encoder = nn.Embedding(self.vocab_size, self.num_embed)
            self.dense1 = nn.Dense(num_hidden, activation='relu', flatten=True)
            self.dense2 = nn.Dense(num_hidden, activation='relu', flatten=True)
            self.dense3 = nn.Dense(vocab_size, flatten=True)

    def forward(self, inputs):
        emd = self.encoder(inputs)
        #print(emd.shape)
        #since the input is shape(batch_size,input(3 characters))
        # we need to extract 0th,1st,2nd character from each batch
        character1 = emd[:, 0, :]
        character2 = emd[:, 1, :]
        character3 = emd[:, 2, :]
        c1_hidden = self.dense1(
            character1)  # green arrow in diagram for character 1
        c2_hidden = self.dense1(
            character2)  # green arrow in diagram for character 2
        c3_hidden = self.dense1(
            character3)  # green arrow in diagram for character 3
        c1_hidden_2 = self.dense2(c1_hidden)  # yellow arrow in diagram
        addition_result = F.add(c2_hidden, c1_hidden_2)  # Total c1 + c2
        addition_hidden = self.dense2(addition_result)  # the yellow arrow
        addition_result_2 = F.add(addition_hidden, c3_hidden)  # Total c1 + c2
        final_output = self.dense3(addition_result_2)
        return final_output


vocab_size = len(chars) + 1  # the vocabsize
num_embed = 30
num_hidden = 256
#model creatings
simple_model = UnRolledRNN_Model(vocab_size, num_embed, num_hidden)
#model initilisation
simple_model.collect_params().initialize(mx.init.Xavier(), ctx=context)
trainer = gluon.Trainer(simple_model.collect_params(), 'adam')
loss = gluon.loss.SoftmaxCrossEntropyLoss()
#sample input shape is of size (2x3)
#output = simple_model(sample_input)
#sample out shape should be(3*87). 87 is our vocab size
#print('the output shape',output.shape)

# %%
#check point file
os.makedirs('checkpoints', exist_ok=True)
filename_unrolled_rnn = "checkpoints/rnn_gluon_abc.params"


# %%
#the actual training
def UnRolledRNNtrain(train_data, label_data, batch_size=32, epochs=10):
    epochs = epochs
    smoothing_constant = .01
    for e in range(epochs):
        for ibatch, i in enumerate(
                range(0, train_data.shape[0] - 1, batch_size)):
            data, target = get_batch(train_data, label_data, i, batch_size)
            data = data.as_in_context(context)
            target = target.as_in_context(context)
            with autograd.record():
                output = simple_model(data)
                L = loss(output, target)
            L.backward()
            trainer.step(data.shape[0])

            ##########################
            #  Keep a moving average of the losses
            ##########################
            if ibatch == 128:
                curr_loss = mx.nd.mean(L).asscalar()
                moving_loss = 0
                moving_loss = (curr_loss if ((i == 0) and (e == 0)) else
                               (1 - smoothing_constant) * moving_loss +
                               (smoothing_constant) * curr_loss)
                print("Epoch %s. Loss: %s, moving_loss %s" % (e, curr_loss,
                                                              moving_loss))
    simple_model.save_parameters(filename_unrolled_rnn)


#%%
epochs = 10
UnRolledRNNtrain(simple_train_data, simple_label_data, batch_size, epochs)

#%%
#loading the model back
simple_model.load_parameters(filename_unrolled_rnn, ctx=context)


#%%
#evaluating the model
def evaluate(input_string):
    idx = [char_indices[c] for c in input_string]
    sample_input = mx.nd.array([[idx[0], idx[1], idx[2]]], ctx=context)
    output = simple_model(sample_input)
    index = mx.nd.argmax(output, axis=1)
    return index.asnumpy()[0]


#%%
#predictions
begin_char = 'lov'
answer = evaluate(begin_char)
print('the predicted answer is ', indices_char[answer])

##

