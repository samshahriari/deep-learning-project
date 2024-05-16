from datasets import CharDataset
from Networks import RNN
from Training import Training

training_dataset = CharDataset('goblet_book.txt', 10)
model = RNN(len(training_dataset.id2char), 64, 1)
training_process = Training(model, 0.001, 5, training_dataset)
training_process.train()
training_process.generate_text(100)