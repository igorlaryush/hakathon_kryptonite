import os
import argparse
from typing import List
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import transforms

from oml import datasets as d
from oml.inference import inference
from oml.models import ViTExtractor
from oml.registry import get_transforms_for_pretrained


# Import custom model wrappers from train.py
class SwinExtractor(torch.nn.Module):
    def __init__(self, model_name="microsoft/swin-tiny-patch4-window7-224"):
        super().__init__()
        from transformers import SwinForImageClassification
        self.model = SwinForImageClassification.from_pretrained(model_name)
        # Remove the classification head, we only need features
        self.embed_dim = self.model.config.hidden_size
        
    def forward(self, x):
        # Extract the hidden states
        outputs = self.model(x, output_hidden_states=True)
        # Use the last hidden state as the embedding
        embeddings = outputs.hidden_states[-1]
        # Global average pooling to get a fixed-size embedding vector
        embeddings = torch.mean(embeddings, dim=1)
        return embeddings
    
    @classmethod
    def from_pretrained(cls, model_name):
        if model_name == "swin":
            model_name = "microsoft/swin-tiny-patch4-window7-224"
        elif model_name == "swin-base":
            model_name = "microsoft/swin-base-patch4-window7-224"
        elif model_name == "swin-large":
            model_name = "microsoft/swin-large-patch4-window7-224"
        
        return cls(model_name)


class ConvNeXtExtractor(torch.nn.Module):
    def __init__(self, model_name="convnext_tiny"):
        super().__init__()
        import torchvision.models as models
        
        # Load the model
        if model_name == "convnext_tiny":
            self.model = models.convnext_tiny(pretrained=True)
            self.embed_dim = 768
        elif model_name == "convnext_small":
            self.model = models.convnext_small(pretrained=True)
            self.embed_dim = 768
        elif model_name == "convnext_base":
            self.model = models.convnext_base(pretrained=True)
            self.embed_dim = 1024
        elif model_name == "convnext_large":
            self.model = models.convnext_large(pretrained=True)
            self.embed_dim = 1536
        
        # Remove the classification head
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        
    def forward(self, x):
        features = self.model(x)
        # Reshape to (batch_size, embedding_dim)
        embeddings = features.squeeze()
        return embeddings
    
    @classmethod
    def from_pretrained(cls, model_name):
        if model_name == "convnext":
            model_name = "convnext_tiny"
        elif model_name == "convnext_small":
            model_name = "convnext_small"
        elif model_name == "convnext_base":
            model_name = "convnext_base"
        elif model_name == "convnext_large":
            model_name = "convnext_large"
            
        return cls(model_name)


class EfficientNetExtractor(torch.nn.Module):
    def __init__(self, model_name="efficientnet_b0"):
        super().__init__()
        import torchvision.models as models
        
        # Load the model
        if model_name == "efficientnet_b0":
            self.model = models.efficientnet_b0(pretrained=True)
            self.embed_dim = 1280
        elif model_name == "efficientnet_b1":
            self.model = models.efficientnet_b1(pretrained=True)
            self.embed_dim = 1280
        elif model_name == "efficientnet_b2":
            self.model = models.efficientnet_b2(pretrained=True)
            self.embed_dim = 1408
        elif model_name == "efficientnet_b3":
            self.model = models.efficientnet_b3(pretrained=True)
            self.embed_dim = 1536
        elif model_name == "efficientnet_b4":
            self.model = models.efficientnet_b4(pretrained=True)
            self.embed_dim = 1792
        elif model_name == "efficientnet_b5":
            self.model = models.efficientnet_b5(pretrained=True)
            self.embed_dim = 2048
        elif model_name == "efficientnet_b6":
            self.model = models.efficientnet_b6(pretrained=True)
            self.embed_dim = 2304
        elif model_name == "efficientnet_b7":
            self.model = models.efficientnet_b7(pretrained=True)
            self.embed_dim = 2560
        
        # Remove the classification head
        self.features = self.model.features
        self.avgpool = self.model.avgpool
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        embeddings = torch.flatten(x, 1)
        return embeddings
    
    @classmethod
    def from_pretrained(cls, model_name):
        if model_name == "efficientnet":
            model_name = "efficientnet_b0"
        elif model_name == "efficientnet_b1":
            model_name = "efficientnet_b1"
        elif model_name == "efficientnet_b2":
            model_name = "efficientnet_b2"
        elif model_name == "efficientnet_b3":
            model_name = "efficientnet_b3"
        elif model_name == "efficientnet_b4":
            model_name = "efficientnet_b4"
        elif model_name == "efficientnet_b5":
            model_name = "efficientnet_b5"
        elif model_name == "efficientnet_b6":
            model_name = "efficientnet_b6"
        elif model_name == "efficientnet_b7":
            model_name = "efficientnet_b7"
            
        return cls(model_name)


# Model wrapper class with embedding head - we need this for properly loading models with heads
class ModelWithHead(torch.nn.Module):
    def __init__(self, base_model, embedding_size=512, num_classes=1000, head_type=None):
        super(ModelWithHead, self).__init__()
        self.base_model = base_model
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.head_type = head_type
        
        # Get the input dimension for the projection head
        if hasattr(base_model, 'embed_dim'):
            self.input_dim = base_model.embed_dim
        else:
            # Default dimension for compatibility
            self.input_dim = 768
            print(f"Warning: Using default embedding dimension: {self.input_dim}")
        
        # Add a projection head if needed for advanced losses
        if head_type in ['arcface', 'cosface', 'adaface']:
            self.head = torch.nn.Linear(self.input_dim, embedding_size, bias=False)
            self.bn = torch.nn.BatchNorm1d(embedding_size)
            self.dropout = torch.nn.Dropout(0.5)
        else:
            self.head = None
    
    def forward(self, x, return_features=False):
        # Get base embeddings from the model
        embeddings = self.base_model(x)
        
        # For inference, we always want the base embeddings
        if return_features or self.head is None:
            return embeddings
        
        # Pass through head for training
        embeddings = self.dropout(embeddings)
        embeddings = self.head(embeddings)
        embeddings = self.bn(embeddings)
        
        return embeddings
    
    # Method to get only embeddings without the head (for inference)
    def get_embeddings(self, x):
        return self.forward(x, return_features=True)


# Wrapper class for base model embeddings (for inference)
class EmbeddingExtractor(torch.nn.Module):
    def __init__(self, model_with_head):
        super().__init__()
        self.model = model_with_head
        
    def forward(self, x):
        return self.model.get_embeddings(x)


# Get advanced augmentation transform (only needed if used during training)
def get_advanced_augmentation_transform(
    img_size=224,
    mean=[0.485, 0.456, 0.406], 
    std=[0.229, 0.224, 0.225]
):
    """Create a transform with advanced augmentations but only those needed for inference."""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    return transform


# Get transforms for ConvNeXt and EfficientNet models
def get_cnn_transforms():
    # Standard transforms for CNN models
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return transform


# Get transforms for Swin models
def get_swin_transforms():
    # Standard transforms for Swin Transformer models
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return transform


def create_sample_sub(pair_ids: List[str], sim_scores: List[float]):
    sub_sim_column = "similarity"
    id_column = "pair_id"
    return pd.DataFrame({id_column: pair_ids, sub_sim_column: sim_scores})


def parse_args():
    parser = argparse.ArgumentParser(description="Create submission file using trained model")
    
    # Model parameters
    parser.add_argument("--model_type", type=str, default="vits16_dino", 
                        help="Model architecture type: vits16_dino, swin, convnext, efficientnet")
    parser.add_argument("--model_path", type=str, default="model.pth", 
                        help="Path to the trained model weights")
    parser.add_argument("--embedding_size", type=int, default=512, 
                        help="Size of embedding for models with heads")
    parser.add_argument("--num_classes", type=int, default=1000,
                        help="Number of classes used during training (for models with heads)")
    parser.add_argument("--head_type", type=str, default=None, 
                        choices=[None, "arcface", "cosface", "adaface"],
                        help="Type of head used during training (if any)")
    
    # Dataset parameters
    parser.add_argument("--test_path", type=str, default="test.csv",
                        help="Path to test CSV file")
    parser.add_argument("--output_path", type=str, default="data/submission.csv",
                        help="Path to save the submission file")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for inference")
    
    # Other parameters
    parser.add_argument("--use_advanced_transforms", action="store_true",
                        help="Use advanced transforms for inference")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    if not os.path.exists("data"):
        os.makedirs("data")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create model based on type
    if "swin" in args.model_type:
        base_model = SwinExtractor.from_pretrained(args.model_type)
        transform = get_swin_transforms() if not args.use_advanced_transforms else get_advanced_augmentation_transform()
        print(f"Using Swin Transformer model: {args.model_type}")
        
    elif "convnext" in args.model_type:
        base_model = ConvNeXtExtractor.from_pretrained(args.model_type)
        transform = get_cnn_transforms() if not args.use_advanced_transforms else get_advanced_augmentation_transform()
        print(f"Using ConvNeXt model: {args.model_type}")
        
    elif "efficientnet" in args.model_type:
        base_model = EfficientNetExtractor.from_pretrained(args.model_type)
        transform = get_cnn_transforms() if not args.use_advanced_transforms else get_advanced_augmentation_transform()
        print(f"Using EfficientNet model: {args.model_type}")
        
    else:
        base_model = ViTExtractor.from_pretrained(args.model_type)
        transform = get_transforms_for_pretrained(args.model_type)[0] if not args.use_advanced_transforms else get_advanced_augmentation_transform()
        print(f"Using ViT model: {args.model_type}")

    # Create model with head if needed
    if args.head_type:
        model = ModelWithHead(
            base_model,
            embedding_size=args.embedding_size,
            num_classes=args.num_classes,
            head_type=args.head_type
        )
        # For inference, we want the model's embeddings without the head
        model = EmbeddingExtractor(model)
        print(f"Created model with {args.head_type} head (using embeddings before the head)")
    else:
        model = base_model
        print("Using base model directly")

    # Load model weights
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    print(f"Loaded model weights from {args.model_path}")

    # Load and prepare test data
    df_test = pd.read_csv(args.test_path)
    test = d.ImageQueryGalleryLabeledDataset(df_test, transform=transform)
    
    # Run inference
    print("Running inference...")
    embeddings = inference(model, test, batch_size=args.batch_size, num_workers=0, verbose=True)

    # Compute similarity scores
    print("Computing similarity scores...")
    e1 = embeddings[::2]
    e2 = embeddings[1::2]
    
    # Fix for the cosine_similarity issue
    sim_scores = torch.nn.functional.cosine_similarity(e1, e2, dim=1).detach().cpu().numpy()

    # Create pair IDs
    pair_ids = df_test["label"].apply(lambda x: f"{x:08d}").to_list()
    pair_ids = pair_ids[::2]  # Take every other ID since we're comparing pairs

    # Create and save submission
    print("Creating submission file...")
    sub_df = create_sample_sub(pair_ids, sim_scores)
    sub_df.to_csv(args.output_path, index=False)
    print(f"Submission saved to {args.output_path}")
