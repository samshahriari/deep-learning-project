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


    ### Test run ###
    

    model_info = {
        'model_type': 'LSTM',
        'num_nodes': 32,
        'context_size': 15,
        'learning_rate': 0.001,
        'num_epochs' : 25
    }


    training_dataset = CharDataset('train.txt', model_info['context_size'], True)
    validation_dataset = CharDataset('val.txt', model_info['context_size'], True)
    test_dataset = CharDataset('test.txt', model_info['context_size'], True)
    if model_info['model_type'] == 'RNN':
        model = RNN(len(training_dataset.id2token), model_info['num_nodes'], len(training_dataset.id2token), num_layers=2)
    else:
        model = LSTM(len(training_dataset.id2token), 32, len(training_dataset.id2token), num_layers=2)
        
    training_process = Training(model, 0.001, False, model_info['n_epochs'], training_dataset, validation_dataset, test_dataset, model_info)
    training_process.train()
    # generated_text = training_process.synthesize_text_char_model(training_process.test_dataset)
    # bleu = bleu_score.calc_BLEU(generated_text, 'train.txt')
    
    #### GRID SEARCH ####
    best_score = float('-inf')
    best_params = None
    
    for params in product(param_grid['node_number'], param_grid['sequence_length']):
        print("Training with parameters:", params)
        training_dataset = CharDataset('goblet_book.txt', params[1], True)
        model = LSTM(len(training_dataset.id2token), params[0], len(training_dataset.id2token), num_layers=2)
        training_process = Training(model, model_info['learning_rate'], False, model_info['n_epochs'], training_dataset, validation_dataset, test_dataset, model_info)
        training_process.train()

        generated_text = training_process.synthesize_text_char_model()

        bleu = bleu_score.calc_BLEU(generated_text, 'train.txt')
        print(f"Parameters: {params}, BLEU score: {bleu}")

        if bleu > best_score:
            best_score = bleu
            best_params = params

    print("Best parameters:", best_params)
    print("Best BLEU score:", best_score)