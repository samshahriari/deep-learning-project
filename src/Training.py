import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# from model import LSTMModel
from datasets import CharDataset
from torch.utils.data import DataLoader
from bleu_score import calc_BLEU, prepare_ref_text
from check_word import read_eng_dictionary, calculate_accuracy
import datetime
import os


#### CHANGE NAME ####
class Training:
    def __init__(self, model, learning_rate, is_word_model, batch_size, number_of_epochs, train_dataset, val_dataset, test_dataset, model_info, plotting):
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.model = model
        self.n = 32
        self.batch_size = batch_size
        self.hidden_size = 64
        self.learning_rate = learning_rate
        self.number_of_epochs = number_of_epochs
        self.training_loader = self.prepare_data(train_dataset)
        self.validation_loader = self.prepare_data(val_dataset)
        self.test_loader = self.prepare_data(test_dataset)
        self.plotting = plotting

        self.is_word_model = is_word_model
        self.chosen_device = self.choose_device()
        self.model.to(self.chosen_device)

        self.model_info = model_info

    ### Rewrite this function ###
    def prepare_data(self, dataset):

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        return loader

    def choose_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Training is runnin on ", device)
        return device

    def train(self):

        eng_words = read_eng_dictionary()
        ref_text = prepare_ref_text('train.txt')

        training_losses = list()
        validation_losses = list()
        word_accuracies = list()
        bleu_scores = list()

        for epoch in range(self.number_of_epochs):
            if self.is_word_model:
                self.synthesize_text_word_model(self.training_loader)
            else:
                self.synthesize_text_char_model(self.training_loader)

            self.model.train()
            h0 = None
            from tqdm import tqdm
            for input_tensor, label in tqdm(self.training_loader):
                # print("input:    ", input_tensor, "label:    ", label)
                # print("input shape:    ", input_tensor.shape, "label shape:    ", label.shape)
                # Move the input and label to the chosen device
                input_tensor, label = input_tensor.to(self.chosen_device), label.to(self.chosen_device)

                # Zero the gradients
                self.optimizer.zero_grad()

                # Forward pass happening here
                # h0 = torch.zeros(1, input_tensor.size(0), self.hidden_size).to(self.chosen_device)
                predictions, _ = self.model(input_tensor, h0)

                train_loss = self.backward_pass(predictions, label)

            train_loss = self.calculate_training_loss()
            print("Epoch", epoch+1, "completed : ", end="")
            print("train_loss=", train_loss, end="")
            val_loss = self.calculate_validation_loss()
            print("val_loss=", val_loss)
            training_losses.append(train_loss)
            validation_losses.append(val_loss)

            folder_path = './saved_models'
            # Create the folder if it doesn't exist
            os.makedirs(folder_path, exist_ok=True)

            # Define the full path for the model file
            model_path = os.path.join(
                folder_path, f"model_{self.model_info['model_type']}_nodes_{self.model_info['num_nodes']}_contextSize_{self.model_info['context_size']}_learningRate_{self.model_info['learning_rate']}_epoch_{epoch+1}.pt")

            # Save the model
            torch.save(self.model.state_dict(), model_path)

            # Kör txt generation efter varje epok istället
            gen_text = ""
            if self.is_word_model:
                gen_text = self.synthesize_text_word_model(self.training_loader)
            else:
                gen_text = self.synthesize_text_char_model(self.training_loader)
            
            bleu = calc_BLEU(gen_text, ref_text)
            correct_words = calculate_accuracy(gen_text, eng_words)

            bleu_scores.append(bleu)
            word_accuracies.append(correct_words)

        if self.plotting:
            self.plot(training_losses, validation_losses, "Training loss", "Validation loss", "Loss", "Epoch", "Training and validation loss over epochs")
            self.plot(bleu_scores, None, "BLEU score", "Epoch", None, "Epoch", "BLEU score over epochs")
            self.plot(word_accuracies, None, "Word accuracy", "Epoch", None, "Epoch", "Word accuracy over epochs")

        ### Test results on test dataset ###
        gen_text = ""
        ref_text = ""
        if self.plotting:
            ref_text = prepare_ref_text('test.txt')
            if self.is_word_model:
                gen_text = self.synthesize_text_word_model(self.test_loader)
            else:
                gen_text = self.synthesize_text_char_model(self.test_loader)

        else:
            ref_text = prepare_ref_text('val.txt')
            if self.is_word_model:
                gen_text = self.synthesize_text_word_model(self.validation_loader)
            else:
                gen_text = self.synthesize_text_char_model(self.validation_loader)

        bleu = calc_BLEU(gen_text, ref_text)
        print("BLEU score on dataset: ", bleu)
        correct_words = calculate_accuracy(gen_text, eng_words)
        print("Word accuracy on dataset: ", correct_words)

        print(training_losses)
        print(validation_losses)
        print(word_accuracies)
        print(gen_text)
        print(bleu)

        return gen_text, validation_losses[-1]

    def calculate_validation_loss(self):
        self.model.eval()
        validation_loss = 0.0
        h0 = None
        with torch.no_grad():
            for input_tensor, label in self.validation_loader:
                input_tensor, label = input_tensor.to(self.chosen_device), label.to(self.chosen_device)
                predictions, _ = self.model(input_tensor, h0)
                loss = self.loss_function(predictions.transpose(1, 2), label)
                validation_loss += loss.item()
        return validation_loss / len(self.validation_loader)

    def calculate_training_loss(self):
        self.model.eval()
        training_loss = 0.0
        h0 = None
        with torch.no_grad():
            for input_tensor, label in self.training_loader:
                input_tensor, label = input_tensor.to(self.chosen_device), label.to(self.chosen_device)
                predictions, _ = self.model(input_tensor, h0)
                loss = self.loss_function(predictions.transpose(1, 2), label)
                training_loss += loss.item()
        return training_loss / len(self.training_loader)

    def plot(self, ylist, ylist2, label, label2, ylabel, xlabel, title):
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(ylist, label=label)
        if ylist2:
            plt.plot(ylist2, label=label2)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # plt.show()
        plt.savefig(
            f"{label}_model_{self.model_info['model_type']}_nodes_{self.model_info['num_nodes']}_contextSize_{self.model_info['context_size']}_learningRate{self.model_info['learning_rate']}_temp_{self.model_info['tn_search']['temperature']}_nucleus_{self.model_info['tn_search']['nucleus_p']}_{timestamp}.png")

    def backward_pass(self, X, y):
        # Calculate the loss
        # Y = torch.unsqueeze(y, 1)
        # Y = Y.expand(-1, X.shape[1])
        # Y = torch.unsqueeze(Y, 2)
        # print(X.shape)
        # print(y.shape)

        loss = self.loss_function(X.transpose(1, 2), y)

        # Backward pass
        loss.backward()

        # Update the weights
        self.optimizer.step()
        return loss.detach().item()

    def sampling(self, logits, temperature=1.0, nucleus=False, nucleus_p=1.0):
        import numpy as np
        import torch.nn.functional as F

        probs = F.softmax(logits / temperature, dim=-1).detach().numpy()
        probs /= np.sum(probs)

        if not nucleus:
            return np.random.choice(np.arange(logits.shape[-1]), p=probs)

        indices = np.argsort(probs)[::-1]
        cum_prob = np.cumsum(probs[indices])

        # Find the number of elements needed to exceed the nucleus_p threshold, using binary search to find the cutoff
        num_elements_needed = np.searchsorted(cum_prob, nucleus_p, side='right') + 1

        num_elements_needed = min(num_elements_needed, len(indices))

        top_v_indices = indices[:num_elements_needed]
        top_probs = probs[top_v_indices]
        top_probs /= np.sum(top_probs)  

        # Sample from the top probabilities
        return np.random.choice(top_v_indices, p=top_probs)

    def synthesize_text_char_model(self, dataloader, n_chars=200):

        gen_text = ""

        # todo: gör detta smidigare :)
        token2id = dataloader.dataset.token2id
        id2token = dataloader.dataset.id2token
        self.model.eval()
        start = "."
        # Add spaces in case the start string is too short
        start = ' '*(self.n-len(start)) + start
        # Ignore everything but the last n characters of the start string
        ids = [token2id[c] for c in start][-self.n:]
        # Generate 200 characters starting from the start string
        try:
            for _ in range(n_chars):

                # Add batch dimension to the input tensor so it can handle more than one input
                input_tensor = torch.tensor(ids).unsqueeze(0).to(self.chosen_device)

                # Get the predictions from the model and remove the batch dimension
                predictions, _ = self.model(input_tensor)
                predictions = predictions.squeeze()[-1].to("cpu")

                # Get the ID of the new character
                new_character_id = self.sampling(predictions, self.model_info['tn_search']['temperature'], self.model_info['tn_search']['nucleus_search'], self.model_info['tn_search']['nucleus_p'])

                # Print the new character
                print(id2token[new_character_id], end='')
                gen_text += id2token[new_character_id]

                # Update the input tensor for the next iteration
                ids.pop(0)

                # Add the new character ID
                ids.append(new_character_id)
            print()

            return gen_text
        except KeyError:
            pass

    def synthesize_text_word_model(self, dataloader, n_words=20):
        gen_text = ""

        # todo: gör detta smidigare :)
        token2id = dataloader.dataset.token2id
        id2token = dataloader.dataset.id2token
        self.model.eval()
        start = "."

        import nltk
        start = nltk.word_tokenize(start)
        # Add spaces in case the start string is too short
        start = ['<pad>']*(self.n-len(start)) + start
        # Ignore everything but the last n characters of the start string
        ids = [token2id[token] for token in start][-self.n:]
        # Generate 200 characters starting from the start string
        print(self.n)

        try:
            for _ in range(n_words):

                # Add batch dimension to the input tensor so it can handle more than one input
                input_tensor = torch.tensor(ids).unsqueeze(0).to(self.chosen_device)
                # Get the predictions from the model and remove the batch dimension
                predictions, _ = self.model(input_tensor)
                predictions = predictions.squeeze()[-1].to("cpu")

                # Get the ID of the new character
                new_character_id = self.sampling(predictions, self.model_info['tn_search']['temperature'], self.model_info['tn_search']['nucleus_search'], self.model_info['tn_search']['nucleus_p'])

                # Print the new character
                print(id2token[new_character_id], end=' ')
                gen_text += id2token[new_character_id] + " "

                # Update the input tensor for the next iteration
                ids.pop(0)

                # Add the new character ID
                ids.append(new_character_id)
            print()

            return gen_text
        except KeyError:
            pass

    def synthesize_text_BPE_model(self, dataloader, n_words=20):
        gen_text_BPE = []

        # todo: gör detta smidigare :)
        tokenizer = dataloader.dataset.tokenizer
        self.model.eval()
        start = "."
        start = ' '*(self.n-len(start)) + start

        # Ignore everything but the last n characters of the start string
        ids = tokenizer.encode(start)[-self.n:]
        # Generate 200 characters starting from the start string
        try:
            for _ in range(n_words):

                # Add batch dimension to the input tensor so it can handle more than one input
                input_tensor = torch.tensor(ids).unsqueeze(0).to(self.chosen_device)

                # Get the predictions from the model and remove the batch dimension
                predictions, _ = self.model(input_tensor)
                predictions = predictions.squeeze()[-1].to("cpu")

                # Get the ID of the new character
                new_character_id = self.sampling(predictions, self.model_info['tn_search']['temperature'], self.model_info['tn_search']['nucleus_search'], self.model_info['tn_search']['nucleus_p'])

                # Print the new character
                gen_text_BPE.append(new_character_id)

                # Update the input tensor for the next iteration
                ids.pop(0)

                # Add the new character ID
                ids.append(new_character_id)
            gen_text = tokenizer.decode(gen_text_BPE)
            # print(gen_text)
            return gen_text
        except KeyError:
            pass
