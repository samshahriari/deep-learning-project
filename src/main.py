from datasets import CharDataset, WordDataset
from Networks import RNN, LSTM
from Training import Training
from itertools import product
import bleu_score

if __name__ == "__main__":
    param_grid = {
        'node_number': [32, 64, 128, 256, 512, 1024],
        'sequence_length': [5, 15, 30, 100],
        'learning_rate': [0.0001, 0.001, 0.01]
    }


    #### RNN ####
    # model_info = {
    #     'model_type': 'RNN',
    #     'num_nodes': 32,
    #     'context_size': 15,
    #     'learning_rate': 0.001,
    #     'num_epochs' : 25
    # }


    # training_dataset = CharDataset('train.txt', model_info['context_size'], True)
    # validation_dataset = CharDataset('val.txt', model_info['context_size'], True)
    # test_dataset = CharDataset('test.txt', model_info['context_size'], True)

    # model = RNN(len(training_dataset.id2token), model_info['num_nodes'], len(training_dataset.id2token))
        
    # training_process = Training(model, model_info['learning_rate'], False, model_info['num_epochs'], training_dataset, validation_dataset, test_dataset, model_info, plotting=True)
    # generated_text, bleu = training_process.train()
    # print("generated_text", generated_text)
    # print("bleu", bleu)
    
    
    ### LSTM ###

    ### Test run ###
    

    model_info = {
        'model_type': 'LSTM',
        'num_nodes': 32,
        'context_size': 15,
        'learning_rate': 0.001,
        'num_epochs' : 8
    }


    # training_dataset = CharDataset('train.txt', model_info['context_size'], True)
    # validation_dataset = CharDataset('val.txt', model_info['context_size'], True)
    # test_dataset = CharDataset('test.txt', model_info['context_size'], True)
    
    
    #### GRID SEARCH ####
    best_score = float('-inf')
    best_params = None
    
    
    for params in product(param_grid['node_number'], param_grid['sequence_length'], param_grid['learning_rate']):
        print("Training with parameters:", params)
        training_dataset = CharDataset('train.txt', params[1], True)
        validation_dataset = CharDataset('val.txt', params[1], True)
        test_dataset = CharDataset('test.txt', params[1], True)

        model = LSTM(len(training_dataset.id2token), params[0], len(training_dataset.id2token), num_layers=2)
        training_process = Training(model, params[2], False, model_info['num_epochs'], training_dataset, validation_dataset, test_dataset, model_info, plotting=False)
        generated_text, bleu = training_process.train()
        
        print(f"Parameters: {params}, BLEU score: {bleu}")

        if bleu > best_score:
            best_score = bleu
            best_params = params

    print("Best parameters:", best_params)
    print("Best BLEU score:", best_score)



    model_info = {
        'model_type': 'LSTM',
        'num_nodes': best_params[0],
        'context_size': best_params[1],
        'learning_rate': best_params[2],
        'num_epochs' : 25
    }

    if model_info['model_type'] == 'RNN':
        model = RNN(len(training_dataset.id2token), model_info['num_nodes'], len(training_dataset.id2token), num_layers=2)
    else:
        model = LSTM(len(training_dataset.id2token), 32, len(training_dataset.id2token), num_layers=2)
        
    training_process = Training(model, model_info['learning_rate'], False, model_info['num_epochs'], training_dataset, validation_dataset, test_dataset, model_info, plotting=True)
    generated_text, bleu = training_process.train()
    print("final generated_text", generated_text)
    print("final bleu", bleu)
    # generated_text = training_process.synthesize_text_char_model(training_process.test_dataset)
    # bleu = bleu_score.calc_BLEU(generated_text, 'train.txt')
    