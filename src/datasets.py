import torch
from torch.utils.data import Dataset

class CharDataset(Dataset):
    id2char = list()
    char2id = dict()

    datapoints = list()
    labels = list()
    def __init__(self, file_path, context_size):
        self.context_size = context_size

        for i in range(len(chars_ids)-n-1):
            self.datapoints.append(chars_ids[i:i+n])
            self.labels.append(chars_ids[i+n])

    def __len__(self):
        return len(self.datapoints)


    def __getitem__(self, idx):
        idx = idx % len(self.datapoints)
        return torch.tensor(self.datapoints[idx]), torch.tensor(self.labels[idx], dtype=torch.long)
    
    def get_ids(self, text):
        restult = list()
        for char in string:
            if char not in self.char2id:
                char2id[char] = len(id2char)
                id2char.append(char)
            restult.append(char2id[char])
        return restult
    


