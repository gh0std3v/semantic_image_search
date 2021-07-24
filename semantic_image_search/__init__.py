#!/usr/bin/env python
# coding: utf-8

# In[3]:


import mygrad as mg
from mygrad.nnet.losses.margin_ranking_loss import margin_ranking_loss
import mynn
import numpy as np
import re, string
from collections import Counter

from mygrad.nnet.initializers import glorot_normal
from mynn.layers.dense import dense
from mynn.optimizers.sgd import SGD
from mynn.optimizers.adam import Adam

from pathlib import Path
import json
import io
import requests
from PIL import Image
from gensim.models import KeyedVectors
import pickle
from collections import defaultdict
import random
import time

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'notebook')


# In[4]:


def download_image(img_url: str) -> Image:
    """Fetches an image from the web.

    Parameters
    ----------
    img_url : string
        The url of the image to fetch.

    Returns
    -------
    PIL.Image
        The image."""

    response = requests.get(img_url)
    return Image.open(io.BytesIO(response.content))


# In[5]:


def tokenize(text):
    """
    
    Parameters 
    ----------
    text: string
    
    
    Returns
    -------
    
    Returns a list containing all the lower-cased versions of the words from the text parameter.
    Each element is a word. All punctuation is removed. 
    Length of list = num words in text
    
    """
    punc_regex = re.compile('[{}]'.format(re.escape(string.punctuation)))
    return punc_regex.sub('', text).lower().split()

def to_vocab(list_of_counters, k=None, stop_words=tuple()):
    """ 
    [word, word, ...] -> sorted list of top-k unique words
    Excludes words included in `stop_words`
    
    Parameters
    ----------
    list_of_counters : Iterable[Iterable[str]]
    
    k : Optional[int]
        If specified, only the top-k words are returned
    
    stop_words : Collection[str]
        A collection of words to be ignored when populating the vocabulary
    """
    # <COGINST>
    vocab = Counter()
    for counter in list_of_counters:
        vocab.update(counter)
        
    for word in set(stop_words):
        vocab.pop(word, None)  # if word not in bag, return None
    return sorted(i for i,j in vocab.most_common(k))
    # </COGINST>

def phrase_idf(phrase_vocab, list_of_counters):
    
    """
    Returns the IDF for the entire phrase. 
   
    """
    N = len(list_of_counters)
   
    # if term i is not in glove, we set nt[i] = N that way its corresponding idf value is 0
    nt = [sum(1 if term in counter else 0 for counter in list_of_counters) for term in phrase_vocab]
    nt = np.array(nt, dtype=float)
    
    nt[nt == 0] = N
    
    
    return np.log10(N / nt)


# In[6]:


class Database:
    def __init__(self):
        self.ID_to_descriptor = resnet18_features
        self.ID_to_URL = dict()
        self.ID_to_captions = defaultdict(list) # key:value | ID --> list of captions corresonding to ID
        self.ID_to_caption_embeddings = defaultdict(list) # key:value | ID --> list of caption embeddings corresonding to ID
        # NOTE: USE "glove" TO CONVERT WORD --> WORD EMBEDDING
        
        # Getting ID --> URL:
        for i in range(len(coco_data["images"])):
            ID = coco_data["images"][i]['id']
            URL = coco_data["images"][i]['coco_url']
            self.ID_to_URL[ID] = URL
        
        # Getting ID --> Captions
        for i in range(len(coco_data["annotations"])):
            ID = coco_data["annotations"][i]['image_id']
            caption = coco_data["annotations"][i]['caption']
            self.ID_to_captions[ID].append(caption)
        
        # Initialize the dataset
        self.make_dataset()
        
        # Shuffle datasets
        self.shuffle_dataset()
        
        # Making List of caption_counters
        all_caption_words = []
        for ID in self.ID_to_captions:
            for caption in self.ID_to_captions[ID]:
                tokenized_caption = set(tokenize(caption))
                for word in tokenized_caption:
                    all_caption_words.append(word)

        all_caption_words = sorted(all_caption_words)
        
        # Calculating all idfs and making word-->idf dict:
        counter = Counter(all_caption_words)
        N = len(coco_data["annotations"])
        words = list(counter.keys())

        nt = list(counter.values())
        nt = np.array(nt, dtype=float)
        all_idfs = np.log10(N / nt)
        self.idf_dict = {k:v for k, v in zip(words, all_idfs)}
        
        # Making ID --> Caption Embeddings
        for i in range(len(coco_data["annotations"])):
            ID = coco_data["annotations"][i]['image_id']
            caption = coco_data["annotations"][i]['caption']
            self.ID_to_caption_embeddings[ID].append(self.parse_query(caption))
    
    # This funciton creates the dataset (only call this once during initialization process)
    def make_dataset(self):
        list_of_IDs = list(self.ID_to_descriptor.keys())
        N = len(list_of_IDs)
        self.dataset = np.zeros((N, 3), dtype=np.int64) # Shape: N, 3
        
        for i in range(N):
            ID = list_of_IDs[i]
            confuser_ID =list_of_IDs[random.randint(0, N-1)]
            while ID == confuser_ID: # Just to make sure that the randomly picked confuser ID isn't the same as the img ID; 1/N chance of happening
                confuser_ID = list_of_IDs[random.randint(0, N-1)]
            caption_index = random.randint(0, len(self.ID_to_captions[ID])-1)
            
            self.dataset[i][0] = caption_index
            self.dataset[i][1] = ID
            self.dataset[i][2] = confuser_ID
    
    '''
    This function randomly shuffles the dataset across its rows (each tuplet)
    and makes the cuts for the training & validation sets;
    call this when you want to shuffle the sets after each epoch.
    '''
    def shuffle_dataset(self):
        np.random.shuffle(self.dataset)
        cut = int(self.dataset.shape[0] * (4/5))
        self.training_set = self.dataset[0:cut]
        self.validation_set = self.dataset[cut:]
    
    # This function parses the query and returns one word embedding that represents the query
    def parse_query(self, phrase):
        phrase_vocab = to_vocab([Counter(tokenize(phrase))])

        glove_embeddings = [(glove[term] if term in glove else np.zeros(200)) for term in phrase_vocab]
        
        idf = [ self.idf_dict[word] if word in self.idf_dict else 0 for word in phrase_vocab ] 
        

       
        w_phrase = sum( glove_embeddings[i] * idf[i] for i in range(len(idf)) )
        

        return w_phrase / np.sqrt((w_phrase ** 2).sum(keepdims=True)) # normalized
    
    # Call this function when model is trained; run it beofre you call "get_k_most_similar_img_URLs"
    def make_img_embeddings(self, model):
        self.img_embeddings = list()
        with mg.no_autodiff:
            for ID in self.ID_to_descriptor.keys():
                self.img_embeddings.append(model(self.ID_to_descriptor[ID]).data)
        self.img_embeddings = np.array(self.img_embeddings)
    
    # Function that takes in a string query and returns k most similar images in database. Returns most disimilar images if similar=False
    def get_k_most_similar_img_URLs(self, query, k=1, similar=True):
        query_embedding = self.parse_query(query)
        results = self.img_embeddings @ query_embedding # (N, 200) dot (200, 1)
        list_of_IDs = list(self.ID_to_descriptor.keys())
        if similar:
            results_sorted_indices = np.argsort(results, axis=0)[::-1]
        else:
            results_sorted_indices = np.argsort(results, axis=0)
        kth_most_similar = results_sorted_indices[0:k].reshape(k,)
        URLs = list()
        captions = list()
        for index in kth_most_similar:
            URLs.append(self.ID_to_URL[list_of_IDs[index]])
            captions.append(self.ID_to_captions[list_of_IDs[index]])
        return URLs, captions
        
    


# In[7]:


def accuracy(true_caption_img_similarity, conf_img_similarity):
    """ Accuracy function for model training"""
    diff = true_caption_img_similarity - conf_img_similarity
    acc = np.mean(diff > 0)
    return acc
    


# In[8]:


class Encoder:
    def __init__(self):
        """ This initializes all of the layers in our model, and sets them
        as attributes of the model. """
        
       
        self.encoder = dense(512, 200, weight_initializer = glorot_normal, bias = False)
        

    def __call__(self, descriptor):
       
        '''Passes data as input to our model, performing a "forward-pass".
        
        This allows us to conveniently initialize a model `m` and then send data through it
        to be classified by calling `m(x)`.
        
        Parameters
        ----------
        x : Union[numpy.ndarray, mygrad.Tensor], shape=(M, D_full)
            A batch of data consisting of M pieces of data,
            each with a dimentionality of D_full.
            
        Returns
        -------
        mygrad.Tensor, shape=(M, D_full)
            The model's prediction for each of the M pieces of data.
        '''
        # keep in mind that this is a linear model - there is no "activation function"
        # involved here
        output = self.encoder(descriptor) 
        return output / np.sqrt((output ** 2).sum(keepdims=True, axis=1)) # normalized output
        # N, 200 --> sum(axis 1) --> (N)
        # N, 200 --> sum(axis 1, kpd=True) -->  broadcastable (N, 1)
        
        
        
    @property
    def parameters(self):
        """ A convenience function for getting all the parameters of our model.
        
        This can be accessed as an attribute, via `model.parameters` 
        
        Returns
        -------
        Tuple[Tensor, ...]
            A tuple containing all of the learnable parameters for our model """
        return self.encoder.parameters
        
        
        

