from datasets import CharDataset, WordDataset
from Networks import RNN, LSTM
from Training import Training
from itertools import product
import bleu_score

if __name__ == "__main__":
    # param_grid = {
    #     'node_number': [128, 512, 1024],
    #     'batch_size': [256, 64, 16],
    #     'learning_rate': [0.0001, 0.001, 0.01]
    # }


    #### RNN ####
    # model_info = {
    #     'model_type': 'RNN',
    #     'num_nodes': 32,
    #     'batch_size': 4,
    #     'context_size': 15,
    #     'learning_rate': 0.001,
    #     'num_epochs' : 25
    # }


    # training_dataset = CharDataset('train.txt', model_info['context_size'], True)
    # validation_dataset = CharDataset('val.txt', model_info['context_size'], True)
    # test_dataset = CharDataset('test.txt', model_info['context_size'], True)

    # model = RNN(len(training_dataset.id2token), model_info['num_nodes'], len(training_dataset.id2token))
        
    # training_process = Training(model, model_info['learning_rate'], False, model_info['batch_size'], model_info['num_epochs'], training_dataset, validation_dataset, test_dataset, model_info, plotting=True)
    # generated_text, bleu = training_process.train()
    # print("generated_text", generated_text)
    # print("bleu", bleu)
    
    
    ### LSTM ###

    ### Test run ###
    

    model_info = {
        'model_type': 'LSTM',
        'num_nodes': 64,
        'learning_rate': 0.001,
        'context_size': 70,
        'num_epochs' : 10,
        'batch_size': 64,
        'WordDataset': False,
        'tn_search': {
            'temperature': 1,
            'nucleus_search': False,
            'nucleus_p': 1
        }
    }

    temp_nucleus_test = {
        'temperatures' : [0.1, 0.5, 0.9 , 2],
        'nuclei' : [0.1, 0.5, 0.9, 0.99],
    }

    training_dataset = CharDataset('train.txt', model_info['context_size'], True)
    validation_dataset = CharDataset('val.txt', model_info['context_size'], True)
    test_dataset = CharDataset('test.txt', model_info['context_size'], True)

    best_temperature = 2
    # for temp in temp_nucleus_test['temperatures']:
    #     model_info['tn_search'] = {
    #         'temperature': temp,
    #         'nucleus_search': False,
    #         'nucleus_p': 1
    #     }

    #     print("temp", temp)
    #     print("nucleus_p", 1)

    #     model = LSTM(len(training_dataset.id2token), model_info['num_nodes'], len(training_dataset.id2token), num_layers=2)
            
    #     training_process = Training(model, model_info['learning_rate'], model_info['WordDataset'], model_info['batch_size'], model_info['num_epochs'], training_dataset, validation_dataset, test_dataset, model_info, plotting=True)
    #     generated_text, val_loss = training_process.train()
    #     print("final generated_text", generated_text)
    #     if val_loss > best_temperature:
    #         best_temperature = val_loss


    
    best_nucleus = 1
    for nucleus_p in temp_nucleus_test['nuclei']:
        model_info['tn_search'] = {
            'temperature': 1,
            'nucleus_search': True,
            'nucleus_p': nucleus_p
        }

        print("temp", 1)
        print("nucleus_p", nucleus_p)
        model = LSTM(len(training_dataset.id2token), model_info['num_nodes'], len(training_dataset.id2token), num_layers=2)
        training_process = Training(model, model_info['learning_rate'], model_info['WordDataset'], model_info['batch_size'], model_info['num_epochs'], training_dataset, validation_dataset, test_dataset, model_info, plotting=True)
        generated_text, val_loss = training_process.train()
        print("final generated_text", generated_text)
        if val_loss > best_nucleus:
            best_nucleus = val_loss

    
    # #### GRID SEARCH ####
    # best_score = float('-inf')
    # best_params = None
    
    
    # for params in product(param_grid['node_number'], param_grid['batch_size'], param_grid['learning_rate']):
    #     print("Training with parameters:", params)
    #     training_dataset = CharDataset('train.txt', model_info['context_size'], True)
    #     validation_dataset = CharDataset('val.txt', model_info['context_size'], True)
    #     test_dataset = CharDataset('test.txt', model_info['context_size'], True)

    #     model = LSTM(len(training_dataset.id2token), params[0], len(training_dataset.id2token), num_layers=2)
    #     training_process = Training(model, params[2], False, params[1], model_info['num_epochs'], training_dataset, validation_dataset, test_dataset, model_info, plotting=False)
    #     generated_text, bleu = training_process.train()
        
    #     print(f"Parameters: {params}, BLEU score: {bleu}")

    #     if bleu > best_score:
    #         best_score = bleu
    #         best_params = params

    # print("Best parameters:", best_params)
    # print("Best BLEU score:", best_score)

    # if best_score < 7:
    #     print("No good parameters found")
    #     exit()



    # model_info = {
    #     'model_type': 'LSTM',
    #     'num_nodes': best_params[0],
    #     'batch_size': best_params[1],
    #     'learning_rate': best_params[2],
    #     'context_size': 32,
    #     'num_epochs' : 25
    # }
    model_info = {
        'model_type': 'LSTM',
        'num_nodes': 64,
        'learning_rate': 0.001,
        'context_size': 70,
        'num_epochs' : 40,
        'batch_size': 64,
        'WordDataset': False,
        'tn_search': {
            'temperature': best_temperature,
            'nucleus_search': True,
            'nucleus_p': best_nucleus
        }
    }


    training_dataset = CharDataset('train.txt', model_info['context_size'], True)
    validation_dataset = CharDataset('val.txt', model_info['context_size'], True)
    test_dataset = CharDataset('test.txt', model_info['context_size'], True)
    

    # if model_info['model_type'] == 'RNN':
    #     model = RNN(len(training_dataset.id2token), model_info['num_nodes'], len(training_dataset.id2token), num_layers=2)
    # else:
    model = LSTM(len(training_dataset.id2token), model_info['num_nodes'], len(training_dataset.id2token), num_layers=2)
        
    training_process = Training(model, model_info['learning_rate'], model_info['WordDataset'], model_info['batch_size'], model_info['num_epochs'], training_dataset, validation_dataset, test_dataset, model_info, plotting=True)
    generated_text, val_loss = training_process.train()
    print("final generated_text", generated_text)
    # generated_text = training_process.synthesize_text_char_model(training_process.test_dataset)
    # bleu = bleu_score.calc_BLEU(generated_text, 'train.txt')
    