import flwr as fl
import torch
from client import LoraClient
from model import get_model_and_tokenizer

print("SCRIPT STARTED: Imports successful!")

# --- 1. DEFINE FUNCTIONS FIRST ---

def aggregate_metrics(metrics):
    accuracies = [m[1]["accuracy"] * m[0] for m in metrics]
    examples = [m[0] for m in metrics]
    
    if sum(examples) == 0:
        return {"accuracy": 0.0}
        
    return {"accuracy": sum(accuracies) / sum(examples)}

def client_fn(cid: str):
    device = torch.device("cpu")
    return LoraClient(client_id=int(cid), device=device).to_client()

# --- 2. RUN MAIN LOGIC ---

if __name__ == "__main__":
    print("ENTERED MAIN: Double underscores are working!")
    print("Downloading/Loading DistilGPT-2 model... (This might take a minute on the first run)")
    
    model, _ = get_model_and_tokenizer()
    initial_params = [val.detach().cpu().numpy() for name, val in model.named_parameters() if val.requires_grad]

    strategy = fl.server.strategy.FedAvg(
        initial_parameters=fl.common.ndarrays_to_parameters(initial_params),
        evaluate_metrics_aggregation_fn=aggregate_metrics, 
    )

    print("STARTING FLOWER SIMULATION...")
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=3, 
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0}, 
    )
