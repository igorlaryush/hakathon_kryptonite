import random
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import mlflow
import os

from oml import datasets as d
from oml.inference import inference
from oml.losses import TripletLossWithMiner
from oml.metrics import calc_retrieval_metrics_rr
from oml.miners import AllTripletsMiner
from oml.models import ViTExtractor
from oml.registry import get_transforms_for_pretrained
from oml.retrieval import RetrievalResults, AdaptiveThresholding
from oml.samplers import BalanceSampler

device = "cuda"
epochs = 10


def fix_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    fix_seed(seed=0)
    
    # Set up MLflow experiment
    experiment_name = "deepfake_detection"
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)
    
    # Start MLflow run
    with mlflow.start_run(run_name="vits16_dino_triplet"):
        model = ViTExtractor.from_pretrained("vits16_dino").to(device).train()
        transform, _ = get_transforms_for_pretrained("vits16_dino")

        df_train, df_val = pd.read_csv("train.csv"), pd.read_csv("val.csv")
        train = d.ImageLabeledDataset(df_train, transform=transform)
        val = d.ImageQueryGalleryLabeledDataset(df_val, transform=transform)

        optimizer = Adam(model.parameters(), lr=1e-4)
        criterion = TripletLossWithMiner(0.1, AllTripletsMiner(), need_logs=True)
        sampler = BalanceSampler(train.get_labels(), n_labels=16, n_instances=4)
        
        # Log parameters
        params = {
            "model": "vits16_dino",
            "optimizer": "Adam",
            "learning_rate": 1e-4,
            "epochs": epochs,
            "triplet_margin": 0.1,
            "miner": "AllTripletsMiner",
            "n_labels": 16,
            "n_instances": 4
        }
        mlflow.log_params(params)

        def training():
            epoch_losses = []

            for epoch in range(epochs):
                pbar = tqdm(DataLoader(train, batch_sampler=sampler))
                pbar.set_description(f"epoch: {epoch}/{epochs}")
                batch_losses = []

                for batch in pbar:
                    embeddings = model(batch["input_tensors"].to(device))
                    loss = criterion(embeddings, batch["labels"].to(device))
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    pbar.set_postfix(criterion.last_logs)
                    batch_losses.append(loss.item())

                avg_epoch_loss = sum(batch_losses) / len(batch_losses)
                epoch_losses.append(avg_epoch_loss)
                print(f"Epoch {epoch} average loss: {avg_epoch_loss:.6f}")
                
                # Log metrics to MLflow
                mlflow.log_metric("train_loss", avg_epoch_loss, step=epoch)
                
                # Log additional metrics from criterion if available
                if hasattr(criterion, 'last_logs'):
                    for key, value in criterion.last_logs.items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(f"train_{key}", value, step=epoch)

            plt.figure(figsize=(10, 5))
            plt.plot(range(1, epochs + 1), epoch_losses, marker='o')
            plt.title('Loss by Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('Average Loss')
            plt.grid(True)
            plt.savefig('loss_by_epoch.png')
            
            # Log the loss plot as an artifact
            mlflow.log_artifact('loss_by_epoch.png')
            plt.show()
            
            return epoch_losses

        def validation():
            embeddings = inference(model, val, batch_size=32, num_workers=0, verbose=True)
            rr = RetrievalResults.from_embeddings(embeddings, val, n_items=10)
            rr = AdaptiveThresholding(n_std=2).process(rr)
            
            # Save visualizations and log them to MLflow
            os.makedirs("validation_results", exist_ok=True)
            query_ids = [2, 1]
            for qid in query_ids:
                fig = rr.visualize(query_ids=[qid], dataset=val, show=False)
                fig_path = f"validation_results/query_{qid}_results.png"
                plt.savefig(fig_path)
                mlflow.log_artifact(fig_path)
                plt.close(fig)
            
            # Also generate and log a visualization with both query IDs
            fig = rr.visualize(query_ids=query_ids, dataset=val, show=True)
            multi_query_path = "validation_results/multi_query_results.png"
            plt.savefig(multi_query_path)
            mlflow.log_artifact(multi_query_path)
            
            results = calc_retrieval_metrics_rr(rr, map_top_k=(10,), cmc_top_k=(1, 5, 10))

            for metric_name in results.keys():
                for k, v in results[metric_name].items():
                    value = v.item()
                    print(f"{metric_name}@{k}: {value}")
                    # Log metrics to MLflow
                    mlflow.log_metric(f"{metric_name}@{k}", value)

        training()
        validation()
        
        # Save model
        torch.save(model.state_dict(), "model.pth")
        mlflow.log_artifact("model.pth")
