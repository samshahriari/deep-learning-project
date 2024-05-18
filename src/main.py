from datasets import CharDataset, WordDataset
from Networks import RNN, LSTM
from Training import Training
import bleu_score


def convert_file_to_string(file):
    with open('goblet_book.txt', 'r') as file:
        txt = file.read().replace('\n', '')
    return txt

def calc_BLEU(gen_txt, ref_txt):
    return bleu_score.calc_bleu_score(gen_txt, ref_txt)


if __name__ == "__main__":
    #todo context window behövs även i training
    training_dataset = CharDataset('../../sprakt/assignment4/exercise/HP_book_1.txt', 5, True, )#'/datasets/dd2417/glove.6B.50d.txt')
    model = LSTM(len(training_dataset.id2token), 64, len(training_dataset.id2token), num_layers=2)#, embedding_size=training_dataset.embedding_dimension, embedding_weights=training_dataset.embeddings)
    training_process = Training(model, 0.001, False, 25, training_dataset)
    training_process.train()

    reference_text = convert_file_to_string('goblet_book.txt')
    generated_text = training_process.synthesize_text_char_model()
    print(calc_BLEU(generated_text, reference_text) * 100)



