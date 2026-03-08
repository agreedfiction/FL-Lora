# FL-Lora: Federated Learning for LLMs

A modular Federated Learning (FL) simulation leveraging the **Flower (flwr)** framework and **PEFT (LoRA)** to fine-tune Large Language Models (DistilGPT-2) in a decentralized, resource-efficient manner.

## 🚀 Overview
This project simulates a federated environment where multiple clients train a global model without sharing private data. By utilizing **Low-Rank Adaptation (LoRA)**, we significantly reduce the number of trainable parameters, allowing for LLM fine-tuning on consumer-grade hardware and edge devices.

## 🛠️ Key Features
- **Privacy-Preserving:** Local training on decentralized data ensures data remains on the client-side.
- **Efficient Fine-Tuning:** Uses Hugging Face's `peft` library for LoRA implementation, reducing memory footprint.
- **Scalable Architecture:** Built on the Flower framework for easy scaling of client simulations.
- **Linux Optimized:** Designed and tested on **Ubuntu/Linux** environments for stable process management.

## 📂 Project Structure
- `main.py`: Orchestrates the Flower server and manages aggregation strategies.
- `client.py`: Handles local model training and communication with the server.
- `model.py`: Defines the LLM architecture (DistilGPT-2) and LoRA configuration.
- `data.py`: Data preprocessing and local dataset partitioning.

## ⚙️ Setup & Installation
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/agreedfiction/FL-Lora.git](https://github.com/agreedfiction/FL-Lora.git)
   cd FL-Lora
