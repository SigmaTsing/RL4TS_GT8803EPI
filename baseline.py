from env import TransformerAgent
import pandas as pd
import numpy as np
import torch
from torch.optim import Adam
from sklearn.preprocessing import MinMaxScaler
import os
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

class TSFDataset(Dataset):
    def __init__(self, raw, input_length, output_length) -> None:
        super().__init__()
        self.x = np.lib.stride_tricks.sliding_window_view(raw, window_shape=(input_length,))[:output_length]
        self.y = np.lib.stride_tricks.sliding_window_view(raw, window_shape=(output_length,))[input_length:]
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

if __name__ == '__main__':
    data_path = '~/Time-LLM/dataset/'
    # raw = pd.read_csv(os.path.join(data_path, 'illness', 'national_illness.csv'))
    raw = pd.read_csv(os.path.join(data_path, 'traffic', 'traffic.csv'))
    
    print('Preprocessing')
    raw = MinMaxScaler((-1, 1)).fit_transform(raw['OT'].to_numpy().reshape((-1, 1))).squeeze()
    input_length = 96
    output_length = 96
    device = 'cuda:0'
    model = TransformerAgent(input_dim= input_length, action_dim=output_length).to(device)
    
    train_span = int(0.8*len(raw))
    train_dataset = TSFDataset(raw[:train_span], input_length, output_length)
    test_dataset = TSFDataset(raw[train_span:], input_length, output_length)
    
    train_loader = DataLoader(train_dataset, batch_size= 64, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size= 64, shuffle = True)
    loss_fn = torch.nn.MSELoss().to(device)
    optimizer = Adam(model.parameters(), lr = 0.001)
    for epoch in (bar:=tqdm(range(1000))):
        for x, y in train_loader:
            action, _ = model(x.to(device).float())
            loss = loss_fn(action, y.to(device).float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if epoch % 50 == 49:
            losses = 0
            for x, y in test_loader:
                action, _ = model(x.to(device).float())
                loss = loss_fn(action, y.to(device).float()).item()
                losses += loss * len(x)
            losses /= len(test_dataset)
            bar.set_description('{:.4f}'.format(losses))
    
        
    
    
    
    