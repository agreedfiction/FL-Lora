import torch
from torch.utils.data import DataLoader, TensorDataset

def load_data(client_id, num_clients=10):
    # This simulates a local dataset for each client
    # In a real project, you'd partition a dataset like WikiText or Alpaca
    torch.manual_seed(client_id)
    
    # Dummy data for testing the pipeline
    x = torch.randint(0, 50256, (100, 32)) # 100 samples, seq_len 32
    y = x.clone()
    
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=8, shuffle=True)