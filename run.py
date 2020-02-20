
# -*- encoding: utf-8 -*-
import os
import ast
import pandas as pd
import logging
import numpy as np

from sklearn.decomposition import PCA
import gensim
from gensim.models import KeyedVectors
import tensorflow as tf
import numpy as np
from gensim.models import Word2Vec
from dotenv import load_dotenv, find_dotenv


# loading model name, language, project path from env
load_dotenv(find_dotenv(), override=True)

project_path = os.environ.get("PROJECT_PATH")
model_name = os.environ.get("MODEL_NAME")
filepath = os.environ.get("FILEPATH")
language = os.environ.get("LANGUAGE")
dimension = 100

class W2vmodel:
    def __init__(self, model_name, language, data, iteration, epoch, windows_size, dimension):
        """

        :param model_name:
        :param language:
        :return:
        """

        self.model_name = model_name
        self.preprocessed_Data = data
        self.language = language
        self.iteration = iteration
        self.epoch = epoch
        self.windows_size = windows_size
        self.trainable_data = []
        self.unique_words = []
        self.dimension = dimension
        self.project_path =project_path 

    def get_list_of_list(self):
        """

        :return:
        """
        self.trainable_data = open(self.preprocessed_Data, "r").read().split("\n")
        self.unique_words  = set([word for each in self.trainable for word in each.split(" ")])


if __name__ == '__main__':
    ss = W2vmodel(model_name=model_name, language=language,data=filepath, iteration=1000, epoch=12, windows_size=5, dimension=dimension)
    
    import pdb
    pdb.set_trace()
    print(ss)
