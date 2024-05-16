import torch
from torch.utils.data import Dataset

class CharDataset(Dataset):

    def __init__(self, file_path, context_size):
        self.context_size = context_size
        self.id2char = []
        self.char2id = {}
        self.datapoints = []
        self.labels = []
        with open(file_path, 'r') as file:
            text = file.read()
            chars_ids = self.get_ids(text)

            for i in range(len(chars_ids) - self.context_size - 1):
                self.datapoints.append(chars_ids[i:i + self.context_size])
                self.labels.append(chars_ids[i+self.context_size])

    def __len__(self):
        return len(self.datapoints)

    def __getitem__(self, idx):
        idx = idx % len(self.datapoints)
        return torch.tensor(self.datapoints[idx]), torch.tensor(self.labels[idx], dtype=torch.long)
    
    def get_ids(self, text):
        restult = list()
        for char in text:
            if char not in self.char2id:
                self.char2id[char] = len(self.id2char)
                self.id2char.append(char)
            restult.append(self.char2id[char])
        return restult
    


