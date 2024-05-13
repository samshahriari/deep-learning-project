import torch
import torch.nn as nn
import torch.optim as optim
# from model import LSTMModel
from datasets import CharDataset
from torch.utils.data import DataLoader


#### CHANGE NAME ####
class Training:
    def __init__(self, model, learning_rate, number_of_epochs):
        # model = LSTMModel(n, hidden_size, len(char_to_id)).to(device)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        self.n = 32
        self.batch_size = 64
        self.hidden_size = 64
        learning_rate = 0.001
        self.number_of_epochs = 5

    ### Rewrite this function ###
    def prepare_data(self, training_loader):

        training_dataset = CharDataset('goblet_book.txt')

        training_loader = DataLoader(training_dataset, batch_size=self.batch_size, collate_fn=char_collate_fn, shuffle=True)

        return training_loader
    
    def choose_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print( "Training is runnin on ", device )
        return device

        

    def train(self):
        training_loader = self.prepare_data()
        chosen_device = self.choose_device()

        self.model.train()
        for epoch in range(self.number_of_epochs):
            for input_tensor, label in training_loader:
                
                # Move the input and label to the chosen device
                input_tensor, label = input_tensor.to(chosen_device), label.to(chosen_device)

                # Zero the gradients
                self.optimizer.zero_grad()
                 
                # Forward pass happening here
                predictions = self.model(input_tensor).to(chosen_device)
                
                loss = self.backward_pass(self, predictions, label)

            print("Epoch", epoch+1, "completed : ", end="")
            print("loss=", loss)

    def backward_pass(self, X, y):
        # Calculate the loss
        loss = self.loss_function(X.squeeze(1), y)
        
        # Backward pass
        loss.backward()

        # Update the weights
        self.optimizer.step()
        return loss.detach().item()
    

    ####### UNFINISHED #######
    def generate_text(self):
        n_chars = 200
        model.eval()
        while True:
            start = input(">")
            if start.strip() == 'quit' :
                break
            # Add spaces in case the start string is too short
            start = ' '*(n-len(start)) + start
            # Ignore everything but the last n characters of the start string
            ids = [char_to_id[c] for c in start][-n:]
            # Generate 200 characters starting from the start string
            try:
                for _ in range(n_chars):
                    input_tensor = torch.tensor(ids).unsqueeze(0).to(device)

                    predictions = model(input_tensor).squeeze().to(device)
                    
                    _, new_character_tensor = predictions.topk(1)
                    
                    new_character_id = new_character_tensor.detach().item()
                    
                    print(id_to_char[new_character_id], end='')
                    
                    ids.pop(0)
                    
                    ids.append(new_character_id)
                print()
            except KeyError:
                continue