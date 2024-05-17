import torch
from torch.utils.data import Dataset
import numpy as np

class CharDataset(Dataset):

    def __init__(self, file_path, context_size):
        self.context_size = context_size
        self.id2token = []
        self.token2id = {}
        self.datapoints = []
        self.labels = []
        with open(file_path, 'r') as file:
            text = file.read()
            chars_ids = self.get_ids(text)

            for i in range(len(chars_ids) - self.context_size - 1):
                self.datapoints.append(chars_ids[i:i + self.context_size])
                self.labels.append(chars_ids[i+1:i+self.context_size+1])
                # self.labels.append(chars_ids[i+self.context_size])
        print("dataset contains", self.__len__() )

    def __len__(self):
        return len(self.datapoints)

    def __getitem__(self, idx):
        idx = idx % len(self.datapoints)
        return torch.tensor(self.datapoints[idx]), torch.tensor(self.labels[idx], dtype=torch.long)
    
    def get_ids(self, text):
        restult = list()
        for char in text:
            if char not in self.token2id:
                self.token2id[char] = len(self.id2token)
                self.id2token.append(char)
            restult.append(self.token2id[char])
        return restult
    

class WordDataset(Dataset):
    
    def load_embeddings(self, embedding_file,
                            pad_token="<pad>", 
                            end_token="<eos>"):
        """
        Reads embeddings from a file.
        Each row is an embedding starting with token and then the vector.
        
        based on DD2417 lab 4
        """
        self.token2id = {}  # Dictionary to store word-to-ID mapping
        self.id2token = []
        self.token2id[pad_token] = 0
        self.token2id[end_token] = 1
        self.id2token.append(pad_token)
        self.id2token.append(end_token)
        self.embeddings = []
        with open(embedding_file, encoding='utf8') as f:
            for line in f:
                data = line.split()
                word = data[0]
                vec = [float(x) for x in data[1:]]
                self.embeddings.append(vec)
                self.token2id[word] = len(self.id2token)
                self.id2token.append(word)
        self.embedding_dimension = len(self.embeddings[0])

        self.embeddings.insert(self.token2id[pad_token], [0]*self.embedding_dimension)  # <pad> has an embedding of just zeros
        self.embeddings.insert(self.token2id[end_token], [-1]*self.embedding_dimension)      # <eos> has an embedding of just minus-ones


    def get_embedding_id(self, token):
        if token in self.token2id:
            return self.token2id[token]
        if token.lower() in self.token2id: # in glove there are just lowercase tokens
            return self.token2id[token.lower()] 

        print(token)
        # generate a random vector for unseen token
        embedding_vector = (np.random.random(self.embedding_dimension)-0.5).tolist() 
        self.embeddings.append(embedding_vector)
        self.token2id[token.lower()] = len(self.id2token)
        self.id2token.append(token.lower())
        return self.token2id[token.lower()] 
        
    def __init__(self, file_path, context_size, embedding_file_path):
        import nltk
        try :
            nltk.word_tokenize("make sure that the nltk vocabulary is already downloaded.")
        except LookupError:
            nltk.download('punkt')
        self.load_embeddings(embedding_file_path)

        self.context_size = context_size
        self.datapoints = []
        self.labels = []
        with open(file_path, "r") as file:
            # nu ser den hela texten som en stor klump och skapar datapunkter mha sliding window men vi kan också se varje rad som en träningspunkt
            tokens = nltk.word_tokenize(file.read())
            for i in range(len(tokens) - self.context_size - 1):
                self.datapoints.append(list(map(self.get_embedding_id, tokens[i:i + self.context_size])))
                self.labels.append(list(map(self.get_embedding_id, tokens[i+1:i+self.context_size+1])))
    
    
    def __len__(self):
        return len(self.datapoints)

    def __getitem__(self, idx):
        idx = idx % len(self.datapoints)
        return torch.tensor(self.datapoints[idx]), torch.tensor(self.labels[idx], dtype=torch.long)