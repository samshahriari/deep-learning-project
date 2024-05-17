from datasets import CharDataset, WordDataset
from Networks import RNN, LSTM
from Training import Training

#todo context window behövs även i training
training_dataset = WordDataset('../../sprakt/assignment4/exercise/HP_book_1.txt', 5, '/datasets/dd2417/glove.6B.50d.txt')
model = LSTM(len(training_dataset.id2token), 64, len(training_dataset.id2token), num_layers=2, embedding_size=training_dataset.embedding_dimension, embedding_weights=training_dataset.embeddings)
training_process = Training(model, 0.001, True, 25, training_dataset)
training_process.train()
