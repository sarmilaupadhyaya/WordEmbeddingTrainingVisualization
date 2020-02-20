import os
import sys

import numpy as np
import tensorflow as tf
from gensim.models import Word2Vec

from gensim.models import KeyedVectors
from tensorboard.plugins import projector

model_path = "/home/ekbana/nep_emb/Nepali_Words_Stemmed_Model_5050.model"
meta_file = "w2vec_metadata.tsv"
    
    
def load_gensim_model(name, test_similarity= None, binary = None):
    """

        :return:
    """
    if binary:
        model = KeyedVectors.load_word2vec_format(name,binary=True)
    else:
        model = KeyedVectors.load(name, mmap='r')

    return model


def visualize_embedding(model, output_path):

    placeholder = np.zeros((len(model.wv.index2word), model.vector_size))
    with open(os.path.join(output_path, meta_file), 'wb') as file_metadata:
        for i, word in enumerate(model.wv.index2word):        
            placeholder[i] = model[word]
            word = word.replace("'", "").replace(",","").strip()
            if len(word) == 1 or word == '':
                print("Ignoring single character or empty line")
                file_metadata.write("Ignored_Word".encode('utf-8') + b'\n')
            else:
                file_metadata.write("{0}".format(word).encode('utf-8') + b'\n')


    session = tf.InteractiveSession()
    embedding_value = tf.Variable(placeholder, trainable=False, name='w2vec_metadata')
    tf.global_variables_initializer().run()
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(output_path, session.graph)

    config = projector.ProjectorConfig()
    embed = config.embeddings.add()
    embed.tensor_name = 'w2vec_metadata'
    embed.metadata_path = meta_file

    projector.visualize_embeddings(writer, config)
    saver.save(session, os.path.join(output_path, 'w2vec_metadata.ckpt'))

if __name__=='__main__':

    if os.path.isdir("tensorboard/"):
        pass
    else:
        os.mkdir("tensorboard")

    model = load_gensim_model(model_path)
    visualize_embedding(model, "tensorboard/")
