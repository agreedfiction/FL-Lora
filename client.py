import flwr as fl
import torch
from collections import OrderedDict
from model import get_model_and_tokenizer
from data import load_data

class LoraClient(fl.client.NumPyClient):
    def __init__(self, client_id, device):
        self.device = device
        self.model, self.tokenizer = get_model_and_tokenizer()
        self.model.to(self.device)
        self.trainloader = load_data(client_id)

    def get_parameters(self, config):
        # We only send parameters that have 'requires_grad=True' (the LoRA layers)
        return [val.cpu().numpy() for name, val in self.model.named_parameters() if val.requires_grad]

    def set_parameters(self, parameters):
        # This puts the aggregated LoRA weights back into the model
        trainable_keys = [name for name, val in self.model.named_parameters() if val.requires_grad]
        params_dict = zip(trainable_keys, parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=False)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
        
        self.model.train()
        for batch in self.trainloader:
            # Batch[0] is input_ids, Batch[1] is labels
            inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
            optimizer.zero_grad()
            outputs = self.model(inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        # Returning dummy metrics to ensure the pipeline completes
        return 0.0, 10, {"accuracy": 0.0}
    
    
