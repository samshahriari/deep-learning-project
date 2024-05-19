from datasets import CharDataset, WordDataset
from Networks import RNN, LSTM
from Training import Training
from nltk.tokenize import word_tokenize
import bleu_score


if __name__ == "__main__":
    #todo context window behövs även i training
    training_dataset = CharDataset('goblet_book.txt', 5, True  )#'/datasets/dd2417/glove.6B.50d.txt')
    model = LSTM(len(training_dataset.id2token), 64, len(training_dataset.id2token), num_layers=2)#, embedding_size=training_dataset.embedding_dimension, embedding_weights=training_dataset.embeddings)
    training_process = Training(model, 0.001, False, 25, training_dataset)
    training_process.train()

    generated_text = training_process.synthesize_text_char_model()

    print("BLEU score:")
    print(bleu_score.calc_BLEU(generated_text, 'goblet_book.txt'))



