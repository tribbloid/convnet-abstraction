#%%
#import necessary dependencies
import mxnet as mx
import numpy as np

#%%
#load cpu context, if using cpu mx.gpu(0)
context = mx.gpu(0)

#%%
# loading https://s3.amazonaws.com/text-datasets/nietzsche.txt nietzsche- You can load anyother text you want (https://cs.stanford.edu/people/karpathy/char-rnn/)
with open("../../data/nlp/nietzsche.txt", errors='ignore') as f:
    text = f.read()
print(len(text))

## %%
#total of characters in dataset
chars = sorted(list(set(text)))
vocab_size = len(chars) + 1
print('total chars:', vocab_size)

#%%
#zeros for padding
chars.insert(0, "\0")

#%%
''.join(chars[1:-6])

#%%
#maps character to unique index e.g. {a:1,b:2....}
char_indices = dict((c, i) for i, c in enumerate(chars))
#maps indices to character (1:a,2:b ....)
indices_char = dict((i, c) for i, c in enumerate(chars))

#%%
#mapping the dataset into index
idx = [char_indices[c] for c in text]

#%%
print(len(idx))

#%%
#testing the mapping
''.join(indices_char[i] for i in idx[:70])

#%% [markdown]
# ## Our unrolled RNN
#
# In this model we map 3 inputs to one output. Later we will design rnn with n inputs to n inputs (sequence to sequence)

#%%
#input for neural network( our basic rnn has 3 inputs, n samples)
cs = 3
c1_dat = [idx[i] for i in range(0, len(idx) - 1 - cs, cs)]
c2_dat = [idx[i + 1] for i in range(0, len(idx) - 1 - cs, cs)]
c3_dat = [idx[i + 2] for i in range(0, len(idx) - 1 - cs, cs)]
#the output of rnn network (single vector)
c4_dat = [idx[i + 3] for i in range(0, len(idx) - 1 - cs, cs)]

#%%
#stacking the inputs to form (3 input features )
x1 = np.stack(c1_dat[:-2])
x2 = np.stack(c2_dat[:-2])
x3 = np.stack(c3_dat[:-2])

#%%
# the output (1 X N data points)
y = np.stack(c4_dat[:-2])

#%%
col_concat = np.array([x1, x2, x3])
t_col_concat = col_concat.T
print(t_col_concat.shape)

#%%
# our sample inputs for the model
x1_nd = mx.nd.array(x1)
x2_nd = mx.nd.array(x2)
x3_nd = mx.nd.array(x3)
sample_input = mx.nd.array([[x1[0], x2[0], x3[0]], [x1[1], x2[1], x3[1]]])

simple_train_data = mx.nd.array(t_col_concat)
simple_label_data = mx.nd.array(y)

#%%
#Set the batchsize as 32, so input is of form 32 X 3
#output is 32 X 1
batch_size = 32


def get_batch(source, label_data, i, batch_size=32):
    bb_size = min(batch_size, source.shape[0] - 1 - i)
    data = source[i:i + bb_size]
    target = label_data[i:i + bb_size]
    #print(target.shape)
    return data, target.reshape((-1,))


#%%
test_bat, test_target = get_batch(simple_train_data, simple_label_data, 5,
                                  batch_size)
print(test_bat.shape)
print(test_target.shape)

#%% [markdown]
# <img src="images/unRolled_rnn.png">

##

