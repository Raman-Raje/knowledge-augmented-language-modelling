import os
import time
import config
import logging
import argparse
import tensorflow as tf

from kalm import *
from data_utils import *

from keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics import accuracy_score

logging.basicConfig(filename='app.log', filemode='w',level=logging.DEBUG)

logging.info("Data preprocessed...")

#get the CONLL2003 dataset
x_train = get_data(config.DATA_PATH + "train.txt")
x_test = get_data(config.DATA_PATH + "test.txt")
x_valid = get_data(config.DATA_PATH + "valid.txt")


general_vocab = dict()
general_vocab["_"] = 0
general_vocab["<s>"] = 1
general_vocab["</s>"] = 2

count = 2
for es in x_train:
    for ew in es.split(" "):
        if ew in general_vocab:
            pass
        else:
            count  += 1
            general_vocab[ew] = count

word_ix = general_vocab
ix_word = dict()

for k,v in general_vocab.items():
    ix_word[v] = k

logging.info("wordix and ixword dictionary created...")

train_x = [[word_ix[w] if w in word_ix else 0 for w in sent.split()] for sent in x_train]
test_x = [[word_ix[w] if w in word_ix else 0 for w in sent.split()] for sent in x_test]
val_x = [[word_ix[w] if w in word_ix else 0 for w in sent.split()] for sent in x_valid]

max_sequence_length = 0

for es in train_x:
    if len(es) >= max_sequence_length:
        max_sequence_length = len(es)

logging.info("Maximum sequence length found - {}".format(max_sequence_length))


####################pad sequences######################

train_x_padded = pad_sequences(train_x,padding="post")
test_x_padded = pad_sequences(test_x,maxlen=train_x_padded.shape[1],padding="post")
val_x_padded = pad_sequences(val_x,maxlen=train_x_padded.shape[1],padding="post")

train_y = train_x_padded
test_y = test_x_padded
val_y = val_x_padded

logging.info("Sequence padded...")

# train data
ds = tf.data.Dataset.from_tensor_slices((train_x_padded, train_x_padded))
ds = ds.take(config.TRAIN_DP).shuffle(config.TRAIN_DP).batch(config.BATCH_SIZE)

# test data
ds_test = tf.data.Dataset.from_tensor_slices((test_x_padded, test_x_padded))
ds_test = ds_test.take(config.TEST_DP).shuffle(config.TEST_DP).batch(1)

# # val data
# ds_val = tf.data.Dataset.from_tensor_slices((val_x_padded, val_x_padded))
# ds_val = ds_val.take(config.NUM_DP).shuffle(config.NUM_DP).batch(config.BATCH_SIZE)


################################# loss funciton and chkpts.################

decoder = Decoder(5000,config.embedding_dim,config.dec_units,
                        config.NB_ENTITIES,config.WH_UNITS,config.WE_UNITS)

optimizer = tf.keras.optimizers.Adam()
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 decoder=decoder)


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')

def loss_function(real, pred):
        
    loss_ = tf.keras.losses.sparse_categorical_crossentropy(real,pred)
    return tf.reduce_mean(loss_)


def train_step(inp, targ, dec_hidden):
  loss = 0

  with tf.GradientTape() as tape:

    dec_input = tf.expand_dims([word_ix['<s>']] * config.BATCH_SIZE, 1)

    # Teacher forcing - feeding the target as the next input
    for t in range(1, targ.shape[1]):
      # passing enc_output to the decoder

      dec_hidden,predictions,_ = decoder(dec_input,dec_hidden)

      loss += loss_function(targ[:, t], predictions)

      # using teacher forcing
      dec_input = tf.expand_dims(targ[:, t], 1)
    
  
  batch_loss = (loss / int(targ.shape[1]))
    

  grads = tape.gradient(loss, decoder.trainable_variables)
  optimizer.apply_gradients(zip(grads, decoder.trainable_variables))
    
  
  return batch_loss



######################training process###################

steps_per_epoch = 10

for epoch in range(config.EPOCHS):
  start = time.time()

  dec_hidden = decoder.initialize_hidden_state(config.BATCH_SIZE)
  total_loss = 0

  print("*"*60)
  print("epoch : ",epoch, " started")
  for (batch, (inp, targ)) in enumerate(ds):
    
    batch_loss = train_step(inp, targ, dec_hidden)
    total_loss += batch_loss
    
  print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                      total_loss / steps_per_epoch))
  print('Time taken for 1 epoch {} sec'.format(time.time() - start))

  print("epoch : ",epoch, " ended\n")



##########################Evaluate################################

def bleu(sentence,result):

    reference = sentence.split(" ")
    candidate = result.split(" ")

    return (sentence_bleu(reference, candidate))


def evaluate(sentence):
    
    dec_hidden = decoder.initialize_hidden_state(1)
    print(sentence)
    dec_input = tf.expand_dims([word_ix['<s>']],0)

    result = ""
    predict = []

    logging.info("Evaluation")
    
    for t in range(max_sequence_length):
        
        dec_hidden,predictions,type_prob = decoder(dec_input,dec_hidden)

        predicted_id = tf.argmax(predictions[0]).numpy()
        predict.append(predicted_id)
        print(predictions.shape)
        print(predicted_id)
        result += ix_word[predicted_id] + ' '

        if ix_word[predicted_id] == '</s>':
            return result, sentence

        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)
    logging.debug("result - {}".format(result))

    return result, sentence,predict


def predict_acc(ds):

    predictions = []

    for batch,(inp, targ) in enumerate(ds):

        print(inp.numpy())
        _,_,pred = evaluate(inp)

        predictions.append(pred)

    return accuracy_score(targ,predictions)



print("Test accuracy :- ",predict_acc(ds_test))

  
