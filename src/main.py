from datasets import CharDataset, WordDataset
from Networks import RNN, LSTM
from Training import Training

#todo context window behövs även i training
training_dataset = WordDataset('goblet_book.txt', 5, '/datasets/dd2417/glove.6B.50d.txt')
model = LSTM(len(training_dataset.id2token), 64, len(training_dataset.id2token), embedding_size=training_dataset.embedding_dimension, embedding_weights=training_dataset.embeddings)
training_process = Training(model, 0.001, 5, training_dataset)
training_process.train()
training_process.generate_text_word_model(100)