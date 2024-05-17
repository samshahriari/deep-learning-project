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
    


