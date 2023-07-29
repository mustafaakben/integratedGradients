import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras

### Utility Functions 
def clear_console():
    """
    Function to clear the console for linux.
    """
    os.system('clear')

# Call the function to clear the console
clear_console()

def get_integrated_gradients(input_tensor, baseline_tensor, model, target_class_idx, m_steps=50):
    """
    Function to calculate integrated gradients for given inputs.
    """
    # Generate alphas
    alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps+1)

    # Initialize TensorArray outside loop to collect gradients.    
    gradient_batches = tf.TensorArray(tf.float32, size=m_steps+1)
    
    # Iterate alphas range and compute gradients
    for alpha in tf.range(len(alphas)):
        # Generate interpolated inputs between baseline and input.
        interpolated_input = baseline_tensor + alphas[alpha] * (input_tensor - baseline_tensor)

        # Compute gradients between model outputs and interpolated inputs.
        with tf.GradientTape() as tape:
            tape.watch(interpolated_input)
            logits = model(interpolated_input)

        gradients = tape.gradient(logits, interpolated_input)
        
        # Print the gradients
        print(f"Gradients at alpha={alphas[alpha]:.3f}: {tf.reduce_mean(gradients):.3f}")

        # Write batch indices and gradients to extend TensorArray.
        gradient_batches = gradient_batches.write(alpha, gradients)
    
    # Stack path gradients together row-wise into single tensor.
    total_gradients = gradient_batches.stack()

    # Integral approximation through averaging gradients.
    avg_gradients = tf.reduce_mean(total_gradients, axis=0)

    # Scale integrated gradients with respect to input.
    integrated_gradients = (input_tensor - baseline_tensor) * avg_gradients

    return integrated_gradients

class CustomEmbedding(layers.Layer):
    """
    Custom Embedding Layer that handles gradient w.r.t. input tensor when the inputs are one-hot encoded.
    """
    def __init__(self, input_dim, output_dim, embeddings_initializer='uniform',
                 embeddings_regularizer=None, embeddings_constraint=None, **kwargs):
        super(CustomEmbedding, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embeddings_initializer = tf.keras.initializers.get(embeddings_initializer)
        self.embeddings_regularizer = tf.keras.regularizers.get(embeddings_regularizer)
        self.embeddings_constraint = tf.keras.constraints.get(embeddings_constraint)

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            shape=(self.input_dim, self.output_dim),
            initializer=self.embeddings_initializer,
            name='embeddings',
            regularizer=self.embeddings_regularizer,
            constraint=self.embeddings_constraint,
            dtype=self.dtype)
        self.built = True

    def call(self, inputs):
        return tf.einsum('nj,jk->nkj',inputs,self.embeddings)

def get_important_words(input_tensor, grads, vocab):
    """
    Function to get the most important words based on the gradients.
    """
    token_indices = np.where(input_tensor==1)[0]
    gradients = grads.numpy()[0]
    gradients = gradients[token_indices]
    ordered_token_id = np.argsort(gradients)[::-1]
    ordered_token_id = token_indices[ordered_token_id]
    return vocab[ordered_token_id] , gradients

# Data Preprocessing
# Use the default parameters to keras.datasets.imdb.load_data
start_char = 1
oov_char = 2
index_from = 3

# Retrieve the training sequences.
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(start_char=start_char, oov_char=oov_char, index_from=index_from)

# Retrieve the word index file mapping words to indices
word_index = keras.datasets.imdb.get_word_index()

# Reverse the word index to obtain a dict mapping indices to words
# And add `index_from` to indices to sync with `x_train`
inverted_word_index = dict((i + index_from, word) for (word, i) in word_index.items())

# Update `inverted_word_index` to include `start_char` and `oov_char`
inverted_word_index[start_char] = "[START]"
inverted_word_index[oov_char] = "[OOV]"

# Decode the sequence in the dataset
train_text = [" ".join(inverted_word_index[i] for i in each) for each in x_train]
test_text = [" ".join(inverted_word_index[i] for i in each) for each in x_test]

# Model Building
MAX_VOCAB = 10000

text_vector = tf.keras.layers.TextVectorization(output_mode='multi_hot',max_tokens=MAX_VOCAB)
text_vector.adapt(train_text)

one_hot_train = text_vector(train_text)
one_hot_test = text_vector(test_text)

# One_Hot - Data
train_data = one_hot_train
test_data = one_hot_test

# Define the model
in_layer = tf.keras.layers.Input(shape=train_data.shape[1:])
embeds = CustomEmbedding(train_data.shape[1], 128)(in_layer)
net = tf.keras.layers.Dense(128,activation='relu')(embeds)
net = tf.keras.layers.Dense(32,activation='relu')(net)
net = tf.keras.layers.Flatten()(net)
out = tf.keras.layers.Dense(1, activation='sigmoid')(net)
model = tf.keras.Model(in_layer, out)
model.summary()

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_data,y_train, epochs=4, batch_size=32, validation_split=0.3)



# Integrated Gradients
VOCABS = np.array(text_vector.get_vocabulary())

# Test the integrated gradients with a random text
baseline= 'random'
baseline_tensor = tf.zeros(shape=(1, MAX_VOCAB)) if baseline == 'zero' else tf.random.uniform(shape=(1, MAX_VOCAB), minval=0, maxval=1)

text_idx = np.random.choice(len(test_text),1)[0]
text = test_text[text_idx]
label = y_test[text_idx]

text = 'This was an amazing moive that I should admit that I did not like it at the very beginning. but I found it very mesmerizing and I would recommend it to everyone.'

input_tensor  = text_vector(text)
grads = get_integrated_gradients(input_tensor, baseline_tensor, model, 1, m_steps=100)

word_importance , _ = get_important_words(input_tensor, grads, VOCABS)

print(text)
print("*"*100)
print("positive" if label==1 else "negative")
print("*"*100)
print(word_importance[:10])
print("*"*100)
