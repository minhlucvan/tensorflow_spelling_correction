#!/usr/bin/env python
# coding: utf-8
# coding=utf-8

# ## Spelling Correction using Tensorflow
# The sections of the project are:
# - Loading the Data
# - Preparing the Data
# - Building the Model
# - Training the Model
# - Fixing Custom Sentences
# - Summary

# In[24]:


# importing the packages
import codecs
import copy
import random

import pandas as pd
import numpy as np
import tensorflow as tf
import os
from os import listdir
from os.path import isfile, join
from collections import namedtuple
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors
import time
import re
from sklearn.model_selection import train_test_split


# ### Loading the Data

# In[23]:


# Method to Load the book from his files
def load_book(path):    
    input_file = os.path.join(path)
    with open(input_file, encoding='utf_8') as f:
        content = f.read()
    return content


# In[3]:


# Collect all of the book file names
path = './books/text/'
book_files = [f for f in listdir(path) if isfile(join(path, f))]
book_files = book_files[1:]


# In[4]:


# Load the books using the file names
books = []
for book in book_files:
    books.append(load_book(path+book))


# In[5]:


# Getting number of words in each book 
for i in range(len(books)):
    print("The number of {} words in book of named {}.".format(len(books[i].split()), book_files[i]))


# In[6]:


# print 500 words of 4th books 
books[4][:500]


# ### Data Preprocessing

# In[7]:


# Method to clean the data
def clean_text(text):    
    text = re.sub(r'\n', ' ', text) 
    text = re.sub(r'[{}@_*>()\\#%+=\[\]]','', text)
    text = re.sub('a0','', text)
    text = re.sub('\'92t','\'t', text)
    text = re.sub('\'92s','\'s', text)
    text = re.sub('\'92m','\'m', text)
    text = re.sub('\'92ll','\'ll', text)
    text = re.sub('\'91','', text)
    text = re.sub('\'92','', text)
    text = re.sub('\'93','', text)
    text = re.sub('\'94','', text)
    text = re.sub('\.','. ', text)
    text = re.sub('\!','! ', text)
    text = re.sub('\?','? ', text)
    text = re.sub(' +',' ', text)
    return text


# In[8]:


# Clean the text of the books
clean_books = []
for book in books:
    clean_books.append(clean_text(book))


# In[9]:


# Check to ensure the text has been cleaned properly
clean_books[0][:500]


# In[10]:


# Create a dictionary to convert the vocabulary (characters) to integers
vocab_to_int = {}
count = 0
for book in clean_books:
    for character in book:
        if character not in vocab_to_int:
            vocab_to_int[character] = count
            count += 1

# Add special tokens to vocab_to_int
codes = ['<PAD>','<EOS>','<GO>']
for code in codes:
    vocab_to_int[code] = count
    count += 1


# In[11]:


# Check the size of vocabulary and all of the values
vocab_size = len(vocab_to_int)
print("The vocabulary contains {} characters.".format(vocab_size))


# In[12]:


# Create another dictionary to convert integers to their respective characters
int_to_vocab = {}
for character, value in vocab_to_int.items():
    int_to_vocab[value] = character


# In[13]:


# Split the text from the books into sentences 
sentences = []
for book in clean_books:
    for sentence in book.split('. '):
        sentences.append(sentence + '.')
print("Total number of sentences are {} .".format(len(sentences)))


# In[14]:


# check for sentence splitting correctly 
sentences[:5]


# In[15]:


# Convert sentences to integers
int_sentences = []

for sentence in sentences:
    int_sentence = []
    for character in sentence:
        int_sentence.append(vocab_to_int[character])
    int_sentences.append(int_sentence)


# In[16]:


# Find the length of each sentence
lengths = []
for sentence in int_sentences:
    lengths.append(len(sentence))
lengths = pd.DataFrame(lengths, columns=["counts"])


# In[17]:


lengths.describe()


# In[18]:


# Limit the data we will use to train our model
max_length = 92
min_length = 10

good_sentences = []

for sentence in int_sentences:
    if len(sentence) <= max_length and len(sentence) >= min_length:
        good_sentences.append(sentence)

print("Total number of {} sentences to train and test our model.".format(len(good_sentences)))


# In[19]:


# Split the data into training and testing sentences
training, testing = train_test_split(good_sentences, test_size = 0.15, random_state = 2)

print("Number of training sentences:", len(training))
print("Number of testing sentences:", len(testing))


# In[20]:


# Sort the sentences by length to reduce padding, which will allow the model to train faster
training_sorted = []
testing_sorted = []

for i in range(min_length, max_length+1):
    for sentence in training:
        if len(sentence) == i:
            training_sorted.append(sentence)
    for sentence in testing:
        if len(sentence) == i:
            testing_sorted.append(sentence)


# In[21]:


# Check to ensure the sentences have been selected and sorted correctly
for i in range(5):
    print(training_sorted[i], len(training_sorted[i]))


# In[22]:

accented_chars_vietnamese = list(map(lambda letters: map(lambda letter: vocab_to_int[letter], letters), [
    ['a', 'á', 'à', 'ả', 'ã', 'ạ', 'â', 'ấ', 'ầ', 'ẩ', 'ẫ', 'ậ', 'ă', 'ắ', 'ằ', 'ẳ', 'ẵ', 'ặ'],
    ['o', 'ó', 'ò', 'ỏ', 'õ', 'ọ', 'ô', 'ố', 'ồ', 'ổ', 'ỗ', 'ộ', 'ơ', 'ớ', 'ờ', 'ở', 'ỡ', 'ợ'],
    ['e', 'é', 'è', 'ẻ', 'ẽ', 'ẹ', 'ê', 'ế', 'ề', 'ể', 'ễ', 'ệ'],
    ['u', 'ú', 'ù', 'ủ', 'ũ', 'ụ', 'ư', 'ứ', 'ừ', 'ử', 'ữ', 'ự'],
    ['i', 'í', 'ì', 'ỉ', 'ĩ', 'ị'],
    ['y', 'ý', 'ỳ', 'ỷ', 'ỹ', 'ỵ'],
    ['d', 'đ'],
]))

accented_chars_vietnamese_all = []

for l in accented_chars_vietnamese:
    accented_chars_vietnamese_all += l

letters = ['a','b','c','d','e','f','g','h','i','j','k','l','m',
           'n','o','p','q','r','s','t','u','v','w','x','y','z',]

def noise_maker0(sentence, threshold):
    noisy_sentence = []
    i = 0
    while i < len(sentence):
        random = np.random.uniform(0, 1, 1)
        # Most characters will be correct since the threshold value is high
        if random < threshold:
            noisy_sentence.append(sentence[i])
        else:
            new_random = np.random.uniform(0, 1, 1)
            # ~33% chance characters will swap locations
            if new_random > 0.67:
                if i == (len(sentence) - 1):
                    # If last character in sentence, it will not be typed
                    continue
                else:
                    # if any other character, swap order with following character
                    noisy_sentence.append(sentence[i + 1])
                    noisy_sentence.append(sentence[i])
                    i += 1
            # ~33% chance an extra lower case letter will be added to the sentence
            elif new_random < 0.33:
                random_letter = np.random.choice(letters, 1)[0]
                noisy_sentence.append(vocab_to_int[random_letter])
                noisy_sentence.append(sentence[i])
            # ~33% chance a character will not be typed
            else:
                pass
        i += 1
    return noisy_sentence

def noise_maker(sentence, threshold):
    noisy_sentence = []
    noisy_set = []

    for index, letter in enumerate(sentence):
        noisy = []
        for set in accented_chars_vietnamese:
            for ch in set:
                if ch == letter:
                    noisy = set

        noisy_set.append(noisy)

    for index, letter in enumerate(sentence):
        char = sentence[index]
        if len(noisy_set[index]) > 0 and random.random() < threshold:
            char_set = noisy_set[index]
            char = random.choice(char_set)
        noisy_sentence.append(char)

    return noisy_sentence


# In[23]:


# Check to ensure noise_maker is making mistakes correctly.
threshold = 0.9
for sentence in training_sorted[:5]:
    print(sentence)
    print(noise_maker(sentence, threshold))
    print()


# ### Building the Model

# In[24]:


# Method to create placeholder
def model_inputs():  
    
    with tf.name_scope('inputs'):
        inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
    with tf.name_scope('targets'):
        targets = tf.placeholder(tf.int32, [None, None], name='targets')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    inputs_length = tf.placeholder(tf.int32, (None,), name='inputs_length')
    targets_length = tf.placeholder(tf.int32, (None,), name='targets_length')
    max_target_length = tf.reduce_max(targets_length, name='max_target_len')

    return inputs, targets, keep_prob, inputs_length, targets_length, max_target_length


# In[25]:


# Remove the last word id from each batch and concat the <GO> to the begining of each batch
def process_encoding_input(targets, vocab_to_int, batch_size):   
    
    with tf.name_scope("process_encoding"):
        ending = tf.strided_slice(targets, [0, 0], [batch_size, -1], [1, 1])
        dec_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), ending], 1)

    return dec_input


# In[26]:


# Create the encoding layer
def encoding_layer(rnn_size, sequence_length, num_layers, rnn_inputs, keep_prob, direction):   
    
    if direction == 1:
        with tf.name_scope("RNN_Encoder_Cell_1D"):
            for layer in range(num_layers):
                with tf.variable_scope('encoder_{}'.format(layer)):
                    lstm = tf.contrib.rnn.LSTMCell(rnn_size)

                    drop = tf.contrib.rnn.DropoutWrapper(lstm, 
                                                         input_keep_prob = keep_prob)

                    enc_output, enc_state = tf.nn.dynamic_rnn(drop, 
                                                              rnn_inputs,
                                                              sequence_length,
                                                              dtype=tf.float32)

            return enc_output, enc_state
        
        
    if direction == 2:
        with tf.name_scope("RNN_Encoder_Cell_2D"):
            for layer in range(num_layers):
                with tf.variable_scope('encoder_{}'.format(layer)):
                    cell_fw = tf.contrib.rnn.LSTMCell(rnn_size)
                    cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, 
                                                            input_keep_prob = keep_prob)

                    cell_bw = tf.contrib.rnn.LSTMCell(rnn_size)
                    cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, 
                                                            input_keep_prob = keep_prob)

                    enc_output, enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, 
                                                                            cell_bw, 
                                                                            rnn_inputs,
                                                                            sequence_length,
                                                                            dtype=tf.float32)
            # Join outputs since we are using a bidirectional RNN
            enc_output = tf.concat(enc_output,2)
            # Use only the forward state because the model can't use both states at once
            return enc_output, enc_state[0]


# In[27]:


# Create the training logits
def training_decoding_layer(dec_embed_input, targets_length, dec_cell, initial_state, output_layer, 
                            vocab_size, max_target_length):    
    
    with tf.name_scope("Training_Decoder"):
        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input,
                                                            sequence_length=targets_length,
                                                            time_major=False)

        training_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                           training_helper,
                                                           initial_state,
                                                           output_layer) 

        training_logits, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                               output_time_major=False,
                                                               impute_finished=True,
                                                               maximum_iterations=max_target_length)
        return training_logits


# In[28]:


# Create the inference logits
def inference_decoding_layer(embeddings, start_token, end_token, dec_cell, initial_state, output_layer,
                             max_target_length, batch_size):    
    
    with tf.name_scope("Inference_Decoder"):
        start_tokens = tf.tile(tf.constant([start_token], dtype=tf.int32), [batch_size], name='start_tokens')

        inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings,
                                                                    start_tokens,
                                                                    end_token)

        inference_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                            inference_helper,
                                                            initial_state,
                                                            output_layer)

        inference_logits, _ = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                                                output_time_major=False,
                                                                impute_finished=True,
                                                                maximum_iterations=max_target_length)

        return inference_logits


# In[29]:


# Create the decoding cell and attention for the training and inference decoding layers
def decoding_layer(dec_embed_input, embeddings, enc_output, enc_state, vocab_size, inputs_length, targets_length, 
                   max_target_length, rnn_size, vocab_to_int, keep_prob, batch_size, num_layers, direction):  
    
    with tf.name_scope("RNN_Decoder_Cell"):
        for layer in range(num_layers):
            with tf.variable_scope('decoder_{}'.format(layer)):
                lstm = tf.contrib.rnn.LSTMCell(rnn_size)
                dec_cell = tf.contrib.rnn.DropoutWrapper(lstm, 
                                                         input_keep_prob = keep_prob)
    
    output_layer = Dense(vocab_size,
                         kernel_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev=0.1))
    
    attn_mech = tf.contrib.seq2seq.BahdanauAttention(rnn_size,
                                                  enc_output,
                                                  inputs_length,
                                                  normalize=False,
                                                  name='BahdanauAttention')
    
    with tf.name_scope("Attention_Wrapper"):
        dec_cell = tf.contrib.seq2seq.DynamicAttentionWrapper(dec_cell,
                                                              attn_mech,
                                                              rnn_size)
    
    initial_state = tf.contrib.seq2seq.DynamicAttentionWrapperState(enc_state,
                                                                    _zero_state_tensors(rnn_size, 
                                                                                        batch_size, 
                                                                                        tf.float32))

    with tf.variable_scope("decode"):
        training_logits = training_decoding_layer(dec_embed_input, 
                                                  targets_length, 
                                                  dec_cell, 
                                                  initial_state,
                                                  output_layer,
                                                  vocab_size, 
                                                  max_target_length)
    with tf.variable_scope("decode", reuse=True):
        inference_logits = inference_decoding_layer(embeddings,  
                                                    vocab_to_int['<GO>'], 
                                                    vocab_to_int['<EOS>'],
                                                    dec_cell, 
                                                    initial_state, 
                                                    output_layer,
                                                    max_target_length,
                                                    batch_size)

    return training_logits, inference_logits


# In[30]:


# Use the previous functions to create the training and inference logits
def seq2seq_model(inputs, targets, keep_prob, inputs_length, targets_length, max_target_length, 
                  vocab_size, rnn_size, num_layers, vocab_to_int, batch_size, embedding_size, direction):    
    
    enc_embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1, 1))
    enc_embed_input = tf.nn.embedding_lookup(enc_embeddings, inputs)
    enc_output, enc_state = encoding_layer(rnn_size, inputs_length, num_layers, 
                                           enc_embed_input, keep_prob, direction)
    
    dec_embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1, 1))
    dec_input = process_encoding_input(targets, vocab_to_int, batch_size)
    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)
    
    training_logits, inference_logits  = decoding_layer(dec_embed_input, 
                                                        dec_embeddings,
                                                        enc_output,
                                                        enc_state, 
                                                        vocab_size, 
                                                        inputs_length, 
                                                        targets_length, 
                                                        max_target_length,
                                                        rnn_size, 
                                                        vocab_to_int, 
                                                        keep_prob, 
                                                        batch_size,
                                                        num_layers,
                                                        direction)
    
    return training_logits, inference_logits


# In[31]:


# Pad sentences with <PAD> so that each sentence of a batch has the same length
def pad_sentence_batch(sentence_batch):    
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [vocab_to_int['<PAD>']] * (max_sentence - len(sentence)) for sentence in sentence_batch]


# In[32]:


# Batch sentences, noisy sentences, and the lengths of their sentences together.With each epoch, sentences will receive new mistakes"
def get_batches(sentences, batch_size, threshold):    
    for batch_i in range(0, len(sentences)//batch_size):
        start_i = batch_i * batch_size
        sentences_batch = sentences[start_i:start_i + batch_size]
        
        sentences_batch_noisy = []
        for sentence in sentences_batch:
            sentences_batch_noisy.append(noise_maker(sentence, threshold))
            
        sentences_batch_eos = []
        for sentence in sentences_batch:
            sentence.append(vocab_to_int['<EOS>'])
            sentences_batch_eos.append(sentence)
            
        pad_sentences_batch = np.array(pad_sentence_batch(sentences_batch_eos))
        pad_sentences_noisy_batch = np.array(pad_sentence_batch(sentences_batch_noisy))
        
        # Need the lengths for the _lengths parameters
        pad_sentences_lengths = []
        for sentence in pad_sentences_batch:
            pad_sentences_lengths.append(len(sentence))
        
        pad_sentences_noisy_lengths = []
        for sentence in pad_sentences_noisy_batch:
            pad_sentences_noisy_lengths.append(len(sentence))
        
        yield pad_sentences_noisy_batch, pad_sentences_batch, pad_sentences_noisy_lengths, pad_sentences_lengths


# In[33]:


# The default parameters
epochs = 100
batch_size = 128
num_layers = 2
rnn_size = 512
embedding_size = 128
learning_rate = 0.0005
direction = 2
threshold = 0.2
keep_probability = 0.75


# In[34]:


# Method to build the graph
def build_graph(keep_prob, rnn_size, num_layers, batch_size, learning_rate, embedding_size, direction):

    tf.reset_default_graph()
    
    # Load the model inputs    
    inputs, targets, keep_prob, inputs_length, targets_length, max_target_length = model_inputs()

    # Create the training and inference logits
    training_logits, inference_logits = seq2seq_model(tf.reverse(inputs, [-1]),
                                                      targets, 
                                                      keep_prob,   
                                                      inputs_length,
                                                      targets_length,
                                                      max_target_length,
                                                      len(vocab_to_int)+1,
                                                      rnn_size, 
                                                      num_layers, 
                                                      vocab_to_int,
                                                      batch_size,
                                                      embedding_size,
                                                      direction)

    # Create tensors for the training logits and inference logits
    training_logits = tf.identity(training_logits.rnn_output, 'logits')

    with tf.name_scope('predictions'):
        predictions = tf.identity(inference_logits.sample_id, name='predictions')
        tf.summary.histogram('predictions', predictions)

    # Create the weights for sequence_loss
    masks = tf.sequence_mask(targets_length, max_target_length, dtype=tf.float32, name='masks')
    
    with tf.name_scope("cost"):
        # Loss function
        cost = tf.contrib.seq2seq.sequence_loss(training_logits, 
                                                targets, 
                                                masks)
        tf.summary.scalar('cost', cost)

    with tf.name_scope("optimze"):
        optimizer = tf.train.AdamOptimizer(learning_rate)

        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)

    # Merge all of the summaries
    merged = tf.summary.merge_all()    

    # Export the nodes 
    export_nodes = ['inputs', 'targets', 'keep_prob', 'cost', 'inputs_length', 'targets_length',
                    'predictions', 'merged', 'train_op','optimizer']
    Graph = namedtuple('Graph', export_nodes)
    local_dict = locals()
    graph = Graph(*[local_dict[each] for each in export_nodes])

    return graph


# ### Training the Model

# In[37]:


# Train the RNN
def train(model, epochs, log_string):   
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Used to determine when to stop the training early
        testing_loss_summary = []

        # Keep track of which batch iteration is being trained
        iteration = 0
        
        display_step = 30 # The progress of the training will be displayed after every 30 batches
        stop_early = 0 
        stop = 3 # If the batch_loss_testing does not decrease in 3 consecutive checks, stop training
        per_epoch = 3 # Test the model 3 times per epoch
        testing_check = (len(training_sorted)//batch_size//per_epoch)-1

        print()
        print("Training Model: {}".format(log_string))

        train_writer = tf.summary.FileWriter('./logs/train/{}'.format(log_string), sess.graph)
        test_writer = tf.summary.FileWriter('./logs/test/{}'.format(log_string))

        for epoch_i in range(1, epochs+1): 
            batch_loss = 0
            batch_time = 0
            
            for batch_i, (input_batch, target_batch, input_length, target_length) in enumerate(
                    get_batches(training_sorted, batch_size, threshold)):
                start_time = time.time()

                summary, loss, _ = sess.run([model.merged,
                                             model.cost, 
                                             model.train_op], 
                                             {model.inputs: input_batch,
                                              model.targets: target_batch,
                                              model.inputs_length: input_length,
                                              model.targets_length: target_length,
                                              model.keep_prob: keep_probability})


                batch_loss += loss
                end_time = time.time()
                batch_time += end_time - start_time

                # Record the progress of training
                train_writer.add_summary(summary, iteration)

                iteration += 1

                if batch_i % display_step == 0 and batch_i > 0:
                    print('Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>6.3f}, Seconds: {:>4.2f}'
                          .format(epoch_i,
                                  epochs, 
                                  batch_i, 
                                  len(training_sorted) // batch_size, 
                                  batch_loss / display_step, 
                                  batch_time))
                    batch_loss = 0
                    batch_time = 0

                #### Testing ####
                if batch_i % testing_check == 0 and batch_i > 0:
                    batch_loss_testing = 0
                    batch_time_testing = 0
                    for batch_i, (input_batch, target_batch, input_length, target_length) in enumerate(
                            get_batches(testing_sorted, batch_size, threshold)):
                        start_time_testing = time.time()
                        summary, loss = sess.run([model.merged,
                                                  model.cost], 
                                                     {model.inputs: input_batch,
                                                      model.targets: target_batch,
                                                      model.inputs_length: input_length,
                                                      model.targets_length: target_length,
                                                      model.keep_prob: 1})

                        batch_loss_testing += loss
                        end_time_testing = time.time()
                        batch_time_testing += end_time_testing - start_time_testing

                        # Record the progress of testing
                        test_writer.add_summary(summary, iteration)

                    n_batches_testing = batch_i + 1
                    print('Testing Loss: {:>6.3f}, Seconds: {:>4.2f}'
                          .format(batch_loss_testing / n_batches_testing, 
                                  batch_time_testing))
                    
                    batch_time_testing = 0

                    # If the batch_loss_testing is at a new minimum, save the model
                    testing_loss_summary.append(batch_loss_testing)
                    if batch_loss_testing <= min(testing_loss_summary):
                        print('New Record!') 
                        stop_early = 0
                        checkpoint = "./{}.ckpt".format(log_string)
                        saver = tf.train.Saver()
                        saver.save(sess, checkpoint)

                    else:
                        print("No Improvement.")
                        stop_early += 1
                        if stop_early == stop:
                            break

            if stop_early == stop:
                print("Stopping Training.")
                break


# In[38]:


# Train the model with the desired tuning parameters
for keep_probability in [0.75]:
    for num_layers in [2]:
        for threshold in [0.95]:
            log_string = 'kp={},nl={},th={}'.format(keep_probability,
                                                    num_layers,
                                                    threshold) 
            model = build_graph(keep_probability, rnn_size, num_layers, batch_size, 
                                learning_rate, embedding_size, direction)
            train(model, epochs, log_string)


# ## Fixing Custom Sentences

# In[39]:


# Prepare the text for the model
def text_to_ints(text):  
    
    text = clean_text(text)
    return [vocab_to_int[word] for word in text]


# ### example01

# In[40]:


# Create your own sentence or use one from the dataset
text = "The first days of her existence in th country were vrey hard for Dolly.."
text = text_to_ints(text)

checkpoint = "./kp=0.75,nl=2,th=0.95.ckpt"

model = build_graph(keep_probability, rnn_size, num_layers, batch_size, learning_rate, embedding_size, direction) 

with tf.Session() as sess:
    # Load saved model
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint)
    
    #Multiply by batch_size to match the model's input parameters
    answer_logits = sess.run(model.predictions, {model.inputs: [text]*batch_size, 
                                                 model.inputs_length: [len(text)]*batch_size,
                                                 model.targets_length: [len(text)+1], 
                                                 model.keep_prob: [1.0]})[0]


# In[41]:


# Remove the padding from the generated sentence
pad = vocab_to_int["<PAD>"] 
# Text
print('\nText')
print('  Input Words: {}'.format("".join([int_to_vocab[i] for i in text])))


# In[42]:


# Summary
print('\nSummary')
print('Response Words: {}'.format("".join([int_to_vocab[i] for i in answer_logits if i != pad])))


# ### example02

# In[43]:


# Create your own sentence or use one from the dataset
text = "Thi is really something impressiv thaat we should look into right away!"
text = text_to_ints(text)

checkpoint = "./kp=0.75,nl=2,th=0.95.ckpt"

model = build_graph(keep_probability, rnn_size, num_layers, batch_size, learning_rate, embedding_size, direction) 

with tf.Session() as sess:
    # Load saved model
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint)
    
    #Multiply by batch_size to match the model's input parameters
    answer_logits = sess.run(model.predictions, {model.inputs: [text]*batch_size, 
                                                 model.inputs_length: [len(text)]*batch_size,
                                                 model.targets_length: [len(text)+1], 
                                                 model.keep_prob: [1.0]})[0]


# In[44]:


# Remove the padding from the generated sentence
pad = vocab_to_int["<PAD>"] 
# Text
print('\nText')
print('  Input Words: {}'.format("".join([int_to_vocab[i] for i in text])))


# In[45]:


# Summary
print('\nSummary')
print('Response Words: {}'.format("".join([int_to_vocab[i] for i in answer_logits if i != pad])))


# ## Summary

# I hope that you have found this project to be rather interesting and useful. The example sentences that I have presented above were specifically chosen, and the model will not always be able to make corrections of this quality. Given the amount of data that we are working with, this model still struggles. For it to be more useful, it would require far more training data, and additional parameter tuning. This parameter values that I have above worked best for me, but I expect there are even better values that I was not able to find.
