
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.autograd import Variable


class LSTMModel(nn.Module):
    def __init__(self, vocab_size = 400001,
                embedding_dim = 50,
                hidden_dim = 256,
                output_dim = 1, 
                n_layers = 2, 
                bidirectional = True, 
                dropout = 0.3,
                embedding_matrix = None,
                batch_first = True,
                device = 'cpu'):
        
        super(LSTMModel, self).__init__()
        
        self.embedding=nn.Embedding(vocab_size, embedding_dim, device=device)
        
        if(embedding_matrix is not None):
            self.embedding.weight = nn.Parameter(embedding_matrix.to(device), requires_grad=False)

        self.rnn=nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, device=device, batch_first=batch_first)

        if(bidirectional):
            self.fc = nn.Linear(hidden_dim*2, output_dim, device=device)
        else:
            self.fc = nn.Linear(hidden_dim, output_dim, device=device)

        self.loss = nn.BCEWithLogitsLoss()

        self.device = device
        self.bidirectional = bidirectional
        self.dropout=nn.Dropout(dropout)
        
    def forward(self,x):
        out=self.embedding(x)
        lstm_out,(hidden,cell)=self.rnn(out)
        if(self.bidirectional):
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else: 
            hidden = hidden[-1,:,:]
        hidden = self.dropout(hidden)

        out=self.fc(hidden.squeeze(0))
        
        return out.squeeze()



def fit(model, data, config, device='cpu'):
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)#,lr=0.001, betas=(0.9,0.999))

    model.train()

    for e in range(config.start_step, config.num_steps+1):
        correct = 0
        n = 0
        for i, (x_batch,y_batch) in enumerate(data):
            x = Variable(x_batch).to(device)
            y = Variable(y_batch).float().to(device)

            optimizer.zero_grad()
            y_pred = model.forward(x)
            loss = model.loss(y_pred, y)

            loss.backward()
            optimizer.step()
            
            predicted = torch.round(F.sigmoid(y_pred))
            
            correct += (predicted == y).sum()
            n += x.shape[0]

            if i % 50 == 0:
                print("-------------------------------------------")
                print(y_pred)
                print(predicted)
                print(y)
                print("{:<15} {:<15} {:<30} {:<30}".format("Epoch: " + str(e), "| Batch: " + str(i), "| Loss: " + str(loss.item()), "| accuracy: " + str(float(correct/n))))
        
        if((e+1) % config.model_save_step == 0):
            torch.save(model.state_dict(), config.model_save_dir + '/lstm-' + str(e+1) + '.pth')


def eval(model, test, config, device = 'cpu'):
    model.eval()
    correct = 0
    n = 0
    for i, (x,y) in enumerate(test):
        x = Variable(x).to(device)
        y = Variable(y).float().to(device)
        y_pred = model.forward(x)
        predicted = torch.round(F.sigmoid(y_pred))
        n += x.shape[0]
        correct +=  (predicted == y).sum().item()
            
    return correct/n 
        
