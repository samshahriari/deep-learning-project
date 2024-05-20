from datasets import CharDataset, WordDataset
from Networks import RNN, LSTM
from Training import Training
from itertools import product
import bleu_score

if __name__ == "__main__":
    param_grid = {
        'node_number': [32, 64, 128, 256, 512, 1024],
        'sequence_length': [5, 15, 30, 100]
    }

    best_score = float('-inf')
    best_params = None

    for params in product(param_grid['node_number'], param_grid['sequence_length']):
        print("Training with parameters:", params)
        training_dataset = CharDataset('goblet_book.txt', params[1], True)
        model = LSTM(len(training_dataset.id2token), params[0], len(training_dataset.id2token), num_layers=2)
        training_process = Training(model, 0.001, False, 25, training_dataset)
        training_process.train()

        generated_text = training_process.synthesize_text_char_model()

        bleu = bleu_score.calc_BLEU(generated_text, 'goblet_book.txt')
        print(f"Parameters: {params}, BLEU score: {bleu}")

        if bleu > best_score:
            best_score = bleu
            best_params = params

    print("Best parameters:", best_params)
    print("Best BLEU score:", best_score)