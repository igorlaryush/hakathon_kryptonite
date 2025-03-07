import random
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader, Sampler
import matplotlib.pyplot as plt
import mlflow
import os
import json
import io
import argparse
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter
from collections import defaultdict
import torchvision.transforms.functional as TF

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


def fix_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ========== Data Augmentation Classes ==========

class RandomErasing(object):
    """Randomly erase a rectangular region from the image."""
    
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, r2=3.3, value=0):
        self.probability = probability
        self.sl = sl  # min erasing area
        self.sh = sh  # max erasing area
        self.r1 = r1  # min aspect ratio
        self.r2 = r2  # max aspect ratio
        self.value = value
        
    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img
            
        img = np.array(img).copy()
        h, w, c = img.shape
        
        # Calculate erasing area
        area = h * w
        target_area = random.uniform(self.sl, self.sh) * area
        aspect_ratio = random.uniform(self.r1, self.r2)
        
        # Calculate dimensions
        h_erased = int(round(np.sqrt(target_area * aspect_ratio)))
        w_erased = int(round(np.sqrt(target_area / aspect_ratio)))
        
        if w_erased < w and h_erased < h:
            x1 = random.randint(0, w - w_erased)
            y1 = random.randint(0, h - h_erased)
            
            if c == 3:
                img[y1:y1+h_erased, x1:x1+w_erased, 0] = self.value
                img[y1:y1+h_erased, x1:x1+w_erased, 1] = self.value
                img[y1:y1+h_erased, x1:x1+w_erased, 2] = self.value
            else:
                img[y1:y1+h_erased, x1:x1+w_erased] = self.value
                
        return Image.fromarray(img)


class GaussianBlur(object):
    """Apply Gaussian Blur to the image with random radius."""
    
    def __init__(self, probability=0.5, min_radius=0.1, max_radius=2.0):
        self.probability = probability
        self.min_radius = min_radius
        self.max_radius = max_radius
        
    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img
            
        radius = random.uniform(self.min_radius, self.max_radius)
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img


class JPEGCompression(object):
    """Simulate JPEG compression artifacts."""
    
    def __init__(self, probability=0.5, quality_min=30, quality_max=90):
        self.probability = probability
        self.quality_min = quality_min
        self.quality_max = quality_max
        
    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img
            
        quality = random.randint(self.quality_min, self.quality_max)
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        compressed_img = Image.open(buffer)
        return compressed_img


class MixUp(object):
    """Apply MixUp to a batch of images and their labels."""
    
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        
    def __call__(self, batch):
        """
        Apply mixup to a batch of images.
        
        Args:
            batch: Dictionary containing 'input_tensors' and 'labels'
            
        Returns:
            Tuple of (mixed_images, mixed_labels_a, mixed_labels_b, lam)
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
            
        batch_size = batch['input_tensors'].size(0)
        
        # Generate permutation indices
        index = torch.randperm(batch_size).to(batch['input_tensors'].device)
        
        # Mix the images
        mixed_input = lam * batch['input_tensors'] + (1 - lam) * batch['input_tensors'][index, :]
        
        # Return mixed images and both sets of labels with lambda
        return {
            'input_tensors': mixed_input,
            'labels': batch['labels'],
            'mixed_labels': batch['labels'][index],
            'lam': lam,
            **{k: v for k, v in batch.items() if k not in ['input_tensors', 'labels']}
        }


class CutMix(object):
    """Apply CutMix to a batch of images and their labels."""
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        
    def __call__(self, batch):
        """
        Apply cutmix to a batch of images.
        
        Args:
            batch: Dictionary containing 'input_tensors' and 'labels'
            
        Returns:
            Tuple of (mixed_images, mixed_labels_a, mixed_labels_b, lam)
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
            
        batch_size = batch['input_tensors'].size(0)
        
        # Generate permutation indices
        index = torch.randperm(batch_size).to(batch['input_tensors'].device)
        
        # Get dimensions
        _, h, w = batch['input_tensors'].size()[-3:]
        
        # Calculate cut size
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)
        
        # Get random center point
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        
        # Calculate box coordinates
        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)
        
        # Create the mixed images
        mixed_input = batch['input_tensors'].clone()
        mixed_input[..., bby1:bby2, bbx1:bbx2] = batch['input_tensors'][index][..., bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda according to the actual cut size
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
        
        # Return mixed images and both sets of labels with lambda
        return {
            'input_tensors': mixed_input,
            'labels': batch['labels'],
            'mixed_labels': batch['labels'][index],
            'lam': lam,
            **{k: v for k, v in batch.items() if k not in ['input_tensors', 'labels']}
        }


# Create advanced augmentation transform
def get_advanced_augmentation_transform(
    img_size=224,
    random_erasing_prob=0.3,
    gaussian_blur_prob=0.3,
    jpeg_compression_prob=0.3,
    mean=[0.485, 0.456, 0.406], 
    std=[0.229, 0.224, 0.225]
):
    """Create a transform with advanced augmentations."""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
        GaussianBlur(probability=gaussian_blur_prob, min_radius=0.1, max_radius=2.0),
        JPEGCompression(probability=jpeg_compression_prob, quality_min=30, quality_max=90),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        transforms.RandomErasing(p=random_erasing_prob, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
    ])
    
    inv_transform = lambda x: transforms.Compose([
        transforms.Normalize(
            mean=[-m/s for m, s in zip(mean, std)],
            std=[1.0/s for s in std]
        ),
    ])(x.clone())
    
    return transform, inv_transform


# Mixing loss for MixUp/CutMix
class MixingLoss(nn.Module):
    """Loss for mixed samples, supports both classification and triplet losses."""
    
    def __init__(self, base_loss, use_mixed_loss=True):
        super().__init__()
        self.base_loss = base_loss
        self.use_mixed_loss = use_mixed_loss
        self.last_logs = {}
        
    def forward(self, embeddings, labels, mixed_labels=None, lam=None, **kwargs):
        # If there's no mixing, just use the base loss
        if mixed_labels is None or lam is None or not self.use_mixed_loss:
            loss = self.base_loss(embeddings, labels, **kwargs)
            if hasattr(self.base_loss, 'last_logs'):
                self.last_logs = self.base_loss.last_logs.copy()
            return loss
        
        # For classification losses (ArcFace, CosFace, AdaFace)
        if hasattr(self.base_loss, 'weight'):
            # Interpolate between the two losses
            loss1 = self.base_loss(embeddings, labels, **kwargs)
            loss2 = self.base_loss(embeddings, mixed_labels, **kwargs)
            loss = lam * loss1 + (1 - lam) * loss2
            
            # Store logs
            if hasattr(self.base_loss, 'last_logs'):
                self.last_logs = self.base_loss.last_logs.copy()
            
            return loss
            
        # For triplet loss, we need to recompute the triplets based on mixed labels
        # For simplicity, we won't apply mixing to triplet loss
        loss = self.base_loss(embeddings, labels, **kwargs)
        if hasattr(self.base_loss, 'last_logs'):
            self.last_logs = self.base_loss.last_logs.copy()
        
        return loss


# Custom Miners for Triplet Loss

class HardestTripletsMiner:
    """Miner that selects hardest triplets in a batch."""
    
    def __init__(self, margin=0.0):
        self.margin = margin
        
    def __call__(self, embeddings, labels):
        # Compute pairwise distances
        dist_matrix = torch.cdist(embeddings, embeddings, p=2)
        
        # For each anchor, find the hardest positive and negative
        triplets = []
        for i, label_i in enumerate(labels):
            # Find positives (same label as anchor)
            pos_indices = torch.where(labels == label_i)[0]
            pos_indices = pos_indices[pos_indices != i]  # Exclude the anchor itself
            
            if len(pos_indices) == 0:
                continue  # Skip if no positives
                
            # Find negatives (different label from anchor)
            neg_indices = torch.where(labels != label_i)[0]
            
            if len(neg_indices) == 0:
                continue  # Skip if no negatives
                
            # Get hardest positive (furthest same-class sample)
            pos_dists = dist_matrix[i, pos_indices]
            hardest_pos_idx = pos_indices[torch.argmax(pos_dists)]
            
            # Get hardest negative (closest different-class sample)
            neg_dists = dist_matrix[i, neg_indices]
            hardest_neg_idx = neg_indices[torch.argmin(neg_dists)]
            
            triplets.append((i, hardest_pos_idx, hardest_neg_idx))
            
        return triplets, {}


class SemiHardTripletsMiner:
    """Miner that selects semi-hard triplets in a batch."""
    
    def __init__(self, margin=0.1):
        self.margin = margin
        
    def __call__(self, embeddings, labels):
        # Compute pairwise distances
        dist_matrix = torch.cdist(embeddings, embeddings, p=2)
        
        # For each anchor, find semi-hard triplets
        triplets = []
        for i, label_i in enumerate(labels):
            # Find positives (same label as anchor)
            pos_indices = torch.where(labels == label_i)[0]
            pos_indices = pos_indices[pos_indices != i]  # Exclude the anchor itself
            
            if len(pos_indices) == 0:
                continue  # Skip if no positives
                
            # Find negatives (different label from anchor)
            neg_indices = torch.where(labels != label_i)[0]
            
            if len(neg_indices) == 0:
                continue  # Skip if no negatives
                
            # For each positive, find semi-hard negatives
            for pos_idx in pos_indices:
                pos_dist = dist_matrix[i, pos_idx]
                
                # Find semi-hard negatives: further than positive but within margin
                neg_dists = dist_matrix[i, neg_indices]
                semi_hard_indices = torch.where((neg_dists > pos_dist) & 
                                               (neg_dists < pos_dist + self.margin))[0]
                
                if len(semi_hard_indices) > 0:
                    # Choose a random semi-hard negative
                    neg_idx = neg_indices[semi_hard_indices[torch.randint(len(semi_hard_indices), (1,))]]
                    triplets.append((i, pos_idx, neg_idx))
                else:
                    # If no semi-hard negative, use the hardest negative
                    hardest_neg_idx = neg_indices[torch.argmin(neg_dists)]
                    triplets.append((i, pos_idx, hardest_neg_idx))
            
        return triplets, {}


# Load deepfake metadata from meta.json
def load_deepfake_metadata(meta_path="data/train/meta.json"):
    """Load metadata about which images are deepfakes."""
    try:
        with open(meta_path, 'r') as f:
            meta_data = json.load(f)
        print(f"Loaded deepfake metadata for {len(meta_data)} images")
        return meta_data
    except Exception as e:
        print(f"Warning: Could not load deepfake metadata: {e}")
        return {}


# Custom dataset that includes deepfake information
class DeepfakeAwareDataset(d.ImageLabeledDataset):
    def __init__(self, df, transform=None, meta_data=None, deepfake_column=None):
        super().__init__(df, transform=transform)
        self.meta_data = meta_data
        self.deepfake_column = deepfake_column
        self.deepfake_labels = []
        
        if meta_data and 'path' in df.columns:
            # Extract deepfake labels from meta_data
            for path in df['path']:
                # Extract the relative path that matches meta.json keys
                rel_path = os.path.basename(os.path.dirname(path)) + "/" + os.path.basename(path)
                # Get deepfake label (1 for deepfake, 0 for real)
                is_deepfake = meta_data.get(rel_path, 0)
                self.deepfake_labels.append(is_deepfake)
            print(f"Extracted deepfake labels for {len(self.deepfake_labels)} images")
            
            # Count real vs synthetic
            real_count = sum(1 for x in self.deepfake_labels if x == 0)
            fake_count = sum(1 for x in self.deepfake_labels if x == 1)
            print(f"Real images: {real_count}, Synthetic images: {fake_count}")
        elif deepfake_column and deepfake_column in df.columns:
            # Use column from DataFrame if specified
            self.deepfake_labels = df[deepfake_column].values
            print(f"Using deepfake labels from column '{deepfake_column}'")
        else:
            # Default to all zeros (assume all real)
            self.deepfake_labels = [0] * len(df)
            print("No deepfake metadata available, assuming all images are real")
    
    def __getitem__(self, idx):
        result = super().__getitem__(idx)
        # Add deepfake label to the result
        result["is_deepfake"] = torch.tensor(self.deepfake_labels[idx], dtype=torch.long)
        return result
    
    def get_deepfake_labels(self):
        """Return list of deepfake labels (1 for fake, 0 for real)."""
        return self.deepfake_labels


# Balanced sampler for real and synthetic data
class DeepfakeBalancedBatchSampler(Sampler):
    """Balanced sampler that ensures each batch has balanced real/fake samples."""
    
    def __init__(self, deepfake_labels, person_labels, n_labels=8, n_instances=4, 
                 real_ratio=0.5, real_label=0, fake_label=1):
        """
        Args:
            deepfake_labels: List of deepfake labels (1 for fake, 0 for real)
            person_labels: List of person/identity labels
            n_labels: Number of person labels per batch
            n_instances: Number of instances per person label
            real_ratio: Target ratio of real images (0.5 means 50% real, 50% fake)
            real_label: Value representing real images in deepfake_labels
            fake_label: Value representing fake images in deepfake_labels
        """
        self.deepfake_labels = deepfake_labels
        self.person_labels = person_labels
        self.n_labels = n_labels
        self.n_instances = n_instances
        self.real_ratio = real_ratio
        self.real_label = real_label
        self.fake_label = fake_label
        
        # Group indices by person label AND real/fake status
        self.label_indices = defaultdict(list)
        
        # Group all indices by real/fake AND person label
        self.real_indices_by_person = defaultdict(list)
        self.fake_indices_by_person = defaultdict(list)
        
        # All person labels
        self.person_label_set = sorted(set(person_labels))
        
        # Find person labels that have both real and fake samples
        self.mixed_person_labels = []
        for i, (person, deepfake) in enumerate(zip(person_labels, deepfake_labels)):
            # Group by combined key
            self.label_indices[(person, deepfake)].append(i)
            
            # Also group by person, separating real and fake
            if deepfake == self.real_label:
                self.real_indices_by_person[person].append(i)
            else:
                self.fake_indices_by_person[person].append(i)
                
            # Track person labels with both real and fake
            if (len(self.real_indices_by_person[person]) > 0 and 
                len(self.fake_indices_by_person[person]) > 0 and
                person not in self.mixed_person_labels):
                self.mixed_person_labels.append(person)
        
        # Calculate sizes
        self.real_persons = [p for p in self.person_label_set 
                            if len(self.real_indices_by_person[p]) >= self.n_instances]
        self.fake_persons = [p for p in self.person_label_set 
                            if len(self.fake_indices_by_person[p]) >= self.n_instances]
        
        print(f"Found {len(self.real_persons)} persons with real images, "
              f"{len(self.fake_persons)} with fake images, "
              f"and {len(self.mixed_person_labels)} with both")
        
        # Calculate number of batches
        target_real_classes = int(self.n_labels * self.real_ratio)
        target_fake_classes = self.n_labels - target_real_classes
        
        max_batches_real = len(self.real_persons) // target_real_classes if target_real_classes > 0 else 0
        max_batches_fake = len(self.fake_persons) // target_fake_classes if target_fake_classes > 0 else 0
        
        # Determine limiting factor
        if max_batches_real == 0:
            self.n_batches = max_batches_fake
            self.real_ratio = 0.0
        elif max_batches_fake == 0:
            self.n_batches = max_batches_real
            self.real_ratio = 1.0
        else:
            self.n_batches = min(max_batches_real, max_batches_fake)
        
        # Adjust if no batches possible
        if self.n_batches == 0:
            print("Warning: Cannot create balanced batches with current settings. Falling back to imbalanced sampling.")
            self.n_batches = max(1, len(self.person_label_set) // self.n_labels)
            
    def __iter__(self):
        # For each batch
        for _ in range(self.n_batches):
            batch_indices = []
            
            # Calculate real/fake split for this batch
            real_classes = int(self.n_labels * self.real_ratio)
            fake_classes = self.n_labels - real_classes
            
            # Sample person IDs for real images
            if real_classes > 0:
                # Randomly sample person labels for real images
                batch_real_persons = random.sample(self.real_persons, real_classes)
                
                # For each selected person, sample instances
                for person in batch_real_persons:
                    # Get indices for this person that are real
                    person_real_indices = self.real_indices_by_person[person]
                    
                    # If we don't have enough, sample with replacement
                    if len(person_real_indices) < self.n_instances:
                        sampled_indices = random.choices(person_real_indices, k=self.n_instances)
                    else:
                        sampled_indices = random.sample(person_real_indices, self.n_instances)
                    
                    batch_indices.extend(sampled_indices)
            
            # Sample person IDs for fake images
            if fake_classes > 0:
                # Randomly sample person labels for fake images
                batch_fake_persons = random.sample(self.fake_persons, fake_classes)
                
                # For each selected person, sample instances
                for person in batch_fake_persons:
                    # Get indices for this person that are fake
                    person_fake_indices = self.fake_indices_by_person[person]
                    
                    # If we don't have enough, sample with replacement
                    if len(person_fake_indices) < self.n_instances:
                        sampled_indices = random.choices(person_fake_indices, k=self.n_instances)
                    else:
                        sampled_indices = random.sample(person_fake_indices, self.n_instances)
                    
                    batch_indices.extend(sampled_indices)
            
            # Yield the batch
            yield batch_indices
    
    def __len__(self):
        return self.n_batches


# Combined loss with deepfake penalty
class DeepfakePenaltyLoss(nn.Module):
    def __init__(self, base_loss, deepfake_weight=1.0):
        super().__init__()
        self.base_loss = base_loss
        self.deepfake_weight = deepfake_weight
        self.last_logs = {}
        
    def forward(self, embeddings, labels, is_deepfake=None):
        # Calculate base loss (identity/similarity)
        base_loss = self.base_loss(embeddings, labels)
        
        # If deepfake information is provided, add penalty
        if is_deepfake is not None and self.deepfake_weight > 0:
            # Convert to float for weighting
            deepfake_float = is_deepfake.float()
            
            # Calculate mean embedding for real and deepfake images
            real_mask = (deepfake_float < 0.5)
            fake_mask = (deepfake_float > 0.5)
            
            # Only apply if we have both real and fake images in batch
            if torch.sum(real_mask) > 0 and torch.sum(fake_mask) > 0:
                real_embeddings = embeddings[real_mask]
                fake_embeddings = embeddings[fake_mask]
                
                # Normalize embeddings
                real_embeddings = F.normalize(real_embeddings, dim=1)
                fake_embeddings = F.normalize(fake_embeddings, dim=1)
                
                # Calculate centroid for real and fake
                real_centroid = torch.mean(real_embeddings, dim=0, keepdim=True)
                fake_centroid = torch.mean(fake_embeddings, dim=0, keepdim=True)
                
                # Penalty: maximize distance between real and fake centroids
                centroid_sim = torch.cosine_similarity(real_centroid, fake_centroid)
                deepfake_penalty = torch.clamp(centroid_sim + 0.3, min=0)  # Margin 0.3
                
                # Individual penalty: make each deepfake further from real centroid
                individual_sims = torch.cosine_similarity(fake_embeddings, real_centroid.expand_as(fake_embeddings))
                individual_penalty = torch.mean(torch.clamp(individual_sims + 0.2, min=0))  # Margin 0.2
                
                # Combined penalty
                penalty = deepfake_penalty + individual_penalty
                
                # Weighted total loss
                total_loss = base_loss + self.deepfake_weight * penalty
                
                # Store logs
                if hasattr(self.base_loss, 'last_logs'):
                    self.last_logs = self.base_loss.last_logs.copy()
                self.last_logs['deepfake_penalty'] = penalty.item()
                
                return total_loss
        
        # If no deepfake info or all same type, just return base loss
        if hasattr(self.base_loss, 'last_logs'):
            self.last_logs = self.base_loss.last_logs.copy()
            
        return base_loss


# Advanced loss functions for embeddings

class ArcFaceLoss(nn.Module):
    def __init__(self, embedding_size, num_classes, scale=30.0, margin=0.5):
        super(ArcFaceLoss, self).__init__()
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.Tensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)
        
    def forward(self, embeddings, labels):
        # Ensure weights are on the same device as embeddings
        if self.weight.device != embeddings.device:
            self.weight = self.weight.to(embeddings.device)
            
        # Normalize weights
        weights_norm = F.normalize(self.weight, dim=1)
        # Normalize embeddings
        embeddings_norm = F.normalize(embeddings, dim=1)
        # Calculate cosine similarity
        cosine = torch.matmul(embeddings_norm, weights_norm.t())
        # Add margin
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        marginal_cosine = torch.cos(theta + self.margin)
        
        # One-hot encode labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        
        # Apply margin to target class only
        output = torch.where(one_hot.bool(), marginal_cosine, cosine)
        # Scale outputs
        output = output * self.scale
        
        return F.cross_entropy(output, labels)


class CosFaceLoss(nn.Module):
    def __init__(self, embedding_size, num_classes, scale=30.0, margin=0.35):
        super(CosFaceLoss, self).__init__()
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.Tensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)
        
    def forward(self, embeddings, labels):
        # Ensure weights are on the same device as embeddings
        if self.weight.device != embeddings.device:
            self.weight = self.weight.to(embeddings.device)
            
        # Normalize weights
        weights_norm = F.normalize(self.weight, dim=1)
        # Normalize embeddings
        embeddings_norm = F.normalize(embeddings, dim=1)
        # Calculate cosine similarity
        cosine = torch.matmul(embeddings_norm, weights_norm.t())
        
        # One-hot encode labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        
        # Apply additive margin to target class only
        output = cosine - one_hot * self.margin
        # Scale outputs
        output = output * self.scale
        
        return F.cross_entropy(output, labels)


class AdaFaceLoss(nn.Module):
    def __init__(self, embedding_size, num_classes, scale=30.0, margin=0.4, h=0.333):
        super(AdaFaceLoss, self).__init__()
        self.scale = scale
        self.margin = margin
        self.h = h
        self.weight = nn.Parameter(torch.Tensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)
        
    def forward(self, embeddings, labels):
        # Ensure weights are on the same device as embeddings
        if self.weight.device != embeddings.device:
            self.weight = self.weight.to(embeddings.device)
            
        # Calculate feature norms
        embeddings_norm = torch.norm(embeddings, dim=1, keepdim=True)
        
        # Normalize weights
        weights_norm = F.normalize(self.weight, dim=1)
        # Normalize embeddings
        embeddings_unit = embeddings / embeddings_norm
        # Calculate cosine similarity
        cosine = torch.matmul(embeddings_unit, weights_norm.t())
        
        # Batch norm stats for adaptive margin
        batch_mean = torch.mean(embeddings_norm)
        batch_std = torch.std(embeddings_norm)
        
        # Calculate adaptive margins
        margin_scaler = (embeddings_norm - batch_mean) / (batch_std + self.h)
        margin_scaler = margin_scaler.clip(-1, 1)
        # Convert to ada_margin
        adaptive_margin = self.margin * (1 + margin_scaler)
        
        # One-hot encode labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        
        # Apply margin to target class only with adaptive scaling
        m_cosine = one_hot * torch.cos(torch.acos(cosine) + adaptive_margin)
        output = torch.where(one_hot.bool(), m_cosine, cosine)
        
        # Scale outputs
        output = output * self.scale
        
        return F.cross_entropy(output, labels)


# Helper function to get embedding dimension for different models
def get_embedding_dim(model):
    """Get the embedding dimension of a model."""
    # For our custom extractors that expose embed_dim
    if hasattr(model, 'embed_dim'):
        return model.embed_dim
    
    # For ViT models from OML, determine based on model name
    # Standard embedding dimensions for ViT variants
    vit_dims = {
        'vits16_dino': 384,
        'vits16': 384,
        'vitb16_dino': 768,
        'vitb16': 768,
        'vitl16_dino': 1024,
        'vitl16': 1024,
    }
    
    # Try to get model name and look up dimension
    if hasattr(model, 'model_name'):
        return vit_dims.get(model.model_name, 768)  # Default to 768 if unknown
    
    # For any other model, we can try a forward pass with a dummy input to determine output size
    try:
        model.eval()  # Set to eval mode for inference
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to(next(model.parameters()).device)
            out = model(dummy_input)
            return out.shape[1]  # Return feature dimension
    except:
        # If all else fails, use 768 as a common dimension
        print("Warning: Could not determine embedding dimension, using default value of 768")
        return 768


# Wrapper class for base model embeddings (for inference)
class EmbeddingExtractor(nn.Module):
    def __init__(self, model_with_head):
        super().__init__()
        self.model = model_with_head
        
    def forward(self, x):
        return self.model.get_embeddings(x)


# Model wrapper class with embedding head for advanced losses
class ModelWithHead(nn.Module):
    def __init__(self, base_model, embedding_size, num_classes, head_type=None):
        super(ModelWithHead, self).__init__()
        self.base_model = base_model
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.head_type = head_type
        
        # Get the input dimension for the projection head
        self.input_dim = get_embedding_dim(base_model)
        print(f"Using embedding dimension: {self.input_dim}")
        
        # Add a projection head if needed for advanced losses
        if head_type in ['arcface', 'cosface', 'adaface']:
            self.head = nn.Linear(self.input_dim, embedding_size, bias=False)
            self.bn = nn.BatchNorm1d(embedding_size)
            self.dropout = nn.Dropout(0.5)
        else:
            self.head = None
    
    def forward(self, x, return_features=False):
        # Get base embeddings from the model
        embeddings = self.base_model(x)
        
        # Pass through head if using advanced loss
        if self.head is not None and not return_features:
            embeddings = self.dropout(embeddings)
            embeddings = self.head(embeddings)
            embeddings = self.bn(embeddings)
        
        return embeddings
    
    # Method to get only embeddings without the head (for inference)
    def get_embeddings(self, x):
        return self.forward(x, return_features=True)


# Wrapper class for Swin Transformer to make it compatible with OML
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


# Wrapper class for ConvNeXt models
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


# Wrapper class for EfficientNet models
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


# Get transforms for ConvNeXt and EfficientNet models
def get_cnn_transforms():
    # Standard transforms for CNN models
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # For compatibility with OML - define inverse transform (not used but needed for interface compatibility)
    inv_transform = lambda x: x
    
    return transform, inv_transform


# Get transforms for Swin models
def get_swin_transforms():
    # Standard transforms for Swin Transformer models
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # For compatibility with OML - define inverse transform (not used but needed for interface compatibility)
    inv_transform = lambda x: x
    
    return transform, inv_transform


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model for deepfake detection")

    # Model parameters
    parser.add_argument("--model", type=str, default="vits16_dino", 
                        help="Model architecture to use: vits16_dino, "
                             "swin/swin-base/swin-large, "
                             "convnext/convnext_small/convnext_base/convnext_large, "
                             "efficientnet/efficientnet_b1/...b7")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    
    # Loss function parameters
    parser.add_argument("--loss", type=str, default="triplet", 
                        choices=["triplet", "arcface", "cosface", "adaface"],
                        help="Loss function to use")
    parser.add_argument("--margin", type=float, default=0.1, help="Margin for loss functions")
    parser.add_argument("--scale", type=float, default=30.0, 
                        help="Scale factor for *Face losses")
    parser.add_argument("--embedding_size", type=int, default=512, 
                        help="Size of embedding for *Face losses")
    
    # Miner parameters
    parser.add_argument("--miner", type=str, default="all",
                      choices=["all", "hardest", "semi_hard"],
                      help="Triplet miner strategy (for triplet loss only)")
    
    # Augmentation parameters
    parser.add_argument("--advanced_augmentation", action="store_true", 
                       help="Use advanced augmentations (color jitter, random erasing, etc.)")
    parser.add_argument("--random_erasing_prob", type=float, default=0.3, 
                       help="Probability of random erasing")
    parser.add_argument("--gaussian_blur_prob", type=float, default=0.3, 
                       help="Probability of applying Gaussian blur")
    parser.add_argument("--jpeg_compression_prob", type=float, default=0.3, 
                       help="Probability of applying JPEG compression")
    parser.add_argument("--mixup", action="store_true", help="Use MixUp augmentation")
    parser.add_argument("--cutmix", action="store_true", help="Use CutMix augmentation")
    parser.add_argument("--mix_alpha", type=float, default=0.2, 
                       help="Alpha parameter for MixUp/CutMix")
    
    # Deepfake parameters
    parser.add_argument("--deepfake_weight", type=float, default=0.0,
                        help="Weight for deepfake penalty (0.0 to disable)")
    parser.add_argument("--meta_path", type=str, default="data/train/meta.json",
                        help="Path to meta.json with deepfake information")
    parser.add_argument("--balance_real_fake", action='store_true',
                        help="Balance real and fake images in each batch")
    parser.add_argument("--real_ratio", type=float, default=0.5,
                        help="Target ratio of real images in each batch (0.5 = 50% real)")

    # Sampler parameters
    parser.add_argument("--n_labels", type=int, default=16, help="Number of labels per batch")
    parser.add_argument("--n_instances", type=int, default=4, help="Number of instances per label")

    # Experiment parameters
    parser.add_argument("--run_name", type=str, default=None, 
                        help="MLflow run name (default: model_name_params)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Use the provided arguments
    fix_seed(seed=args.seed)
    epochs = args.epochs
    
    # Generate a default run name if not provided
    if args.run_name is None:
        deepfake_suffix = f"_df{args.deepfake_weight}" if args.deepfake_weight > 0 else ""
        balance_suffix = "_balanced" if args.balance_real_fake else ""
        miner_suffix = f"_{args.miner}" if args.loss == "triplet" and args.miner != "all" else ""
        aug_suffix = "_advaug" if args.advanced_augmentation else ""
        mix_suffix = "_mixup" if args.mixup else "_cutmix" if args.cutmix else ""
        
        args.run_name = f"{args.model}_{args.loss}{miner_suffix}_m{args.margin}_lr{args.lr}_e{args.epochs}{deepfake_suffix}{balance_suffix}{aug_suffix}{mix_suffix}"

    # Set up MLflow experiment
    experiment_name = "deepfake_detection"
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)

    # Load deepfake metadata if needed
    meta_data = None
    if args.deepfake_weight > 0 or args.balance_real_fake:
        meta_data = load_deepfake_metadata(args.meta_path)

    # Start MLflow run
    with mlflow.start_run(run_name=args.run_name):
        # Model and transform selection
        if "swin" in args.model:
            base_model = SwinExtractor.from_pretrained(args.model).to(device)
            
            if args.advanced_augmentation:
                transform, inv_transform = get_advanced_augmentation_transform(
                    random_erasing_prob=args.random_erasing_prob,
                    gaussian_blur_prob=args.gaussian_blur_prob,
                    jpeg_compression_prob=args.jpeg_compression_prob
                )
                print("Using advanced augmentations for Swin model")
            else:
                transform, inv_transform = get_swin_transforms()
                
            print(f"Using Swin Transformer model: {args.model}")
            
        elif "convnext" in args.model:
            base_model = ConvNeXtExtractor.from_pretrained(args.model).to(device)
            
            if args.advanced_augmentation:
                transform, inv_transform = get_advanced_augmentation_transform(
                    random_erasing_prob=args.random_erasing_prob,
                    gaussian_blur_prob=args.gaussian_blur_prob,
                    jpeg_compression_prob=args.jpeg_compression_prob
                )
                print("Using advanced augmentations for ConvNeXt model")
            else:
                transform, inv_transform = get_cnn_transforms()
                
            print(f"Using ConvNeXt model: {args.model}")
            
        elif "efficientnet" in args.model:
            base_model = EfficientNetExtractor.from_pretrained(args.model).to(device)
            
            if args.advanced_augmentation:
                transform, inv_transform = get_advanced_augmentation_transform(
                    random_erasing_prob=args.random_erasing_prob,
                    gaussian_blur_prob=args.gaussian_blur_prob,
                    jpeg_compression_prob=args.jpeg_compression_prob
                )
                print("Using advanced augmentations for EfficientNet model")
            else:
                transform, inv_transform = get_cnn_transforms()
                
            print(f"Using EfficientNet model: {args.model}")
            
        else:
            base_model = ViTExtractor.from_pretrained(args.model).to(device)
            
            if args.advanced_augmentation:
                transform, inv_transform = get_advanced_augmentation_transform(
                    random_erasing_prob=args.random_erasing_prob,
                    gaussian_blur_prob=args.gaussian_blur_prob,
                    jpeg_compression_prob=args.jpeg_compression_prob
                )
                print("Using advanced augmentations for ViT model")
            else:
                transform, inv_transform = get_transforms_for_pretrained(args.model)
                
            print(f"Using ViT model: {args.model}")

        df_train = pd.read_csv("train.csv")
        df_val = pd.read_csv("val.csv")
        
        # Use custom dataset with deepfake information if needed
        if args.deepfake_weight > 0 or args.balance_real_fake:
            train = DeepfakeAwareDataset(df_train, transform=transform, meta_data=meta_data)
            if args.deepfake_weight > 0:
                print(f"Using deepfake-aware dataset with penalty weight {args.deepfake_weight}")
            if args.balance_real_fake:
                print(f"Using balanced real/fake sampling with real ratio {args.real_ratio}")
        else:
            train = d.ImageLabeledDataset(df_train, transform=transform)
        
        val = d.ImageQueryGalleryLabeledDataset(df_val, transform=transform)
        
        # Get unique classes for *Face losses
        num_classes = len(np.unique(train.get_labels()))
        print(f"Dataset has {num_classes} unique classes")

        # Create miner based on the selected strategy
        if args.loss == "triplet":
            if args.miner == "hardest":
                miner = HardestTripletsMiner(margin=args.margin)
                print(f"Using Hardest Triplets Miner with margin={args.margin}")
            elif args.miner == "semi_hard":
                miner = SemiHardTripletsMiner(margin=args.margin)
                print(f"Using Semi-Hard Triplets Miner with margin={args.margin}")
            else:  # default: "all"
                miner = AllTripletsMiner()
                print(f"Using All Triplets Miner")
        else:
            # For non-triplet losses, create a dummy miner that won't be used
            miner = None

        # Setup base loss function
        if args.loss == 'arcface':
            base_criterion = ArcFaceLoss(args.embedding_size, num_classes, scale=args.scale, margin=args.margin).to(device)
            need_labels_directly = True
            print(f"Using ArcFace loss with margin={args.margin}, scale={args.scale}")
        elif args.loss == 'cosface':
            base_criterion = CosFaceLoss(args.embedding_size, num_classes, scale=args.scale, margin=args.margin).to(device)
            need_labels_directly = True
            print(f"Using CosFace loss with margin={args.margin}, scale={args.scale}")
        elif args.loss == 'adaface':
            base_criterion = AdaFaceLoss(args.embedding_size, num_classes, scale=args.scale, margin=args.margin).to(device)
            need_labels_directly = True
            print(f"Using AdaFace loss with margin={args.margin}, scale={args.scale}")
        else:
            base_criterion = TripletLossWithMiner(args.margin, miner, need_logs=True)
            need_labels_directly = False
            print(f"Using Triplet loss with margin={args.margin}")
        
        # Wrap with MixUp/CutMix if needed
        if args.mixup or args.cutmix:
            criterion = MixingLoss(base_criterion)
            print(f"Using MixingLoss with {'MixUp' if args.mixup else 'CutMix'}, alpha={args.mix_alpha}")
        else:
            criterion = base_criterion
        
        # Wrap with deepfake penalty if needed
        if args.deepfake_weight > 0:
            if hasattr(criterion, 'base_loss'):
                # If we already have a MixingLoss wrapper, update its base_loss
                criterion.base_loss = DeepfakePenaltyLoss(criterion.base_loss, deepfake_weight=args.deepfake_weight)
            else:
                criterion = DeepfakePenaltyLoss(criterion, deepfake_weight=args.deepfake_weight)
            print(f"Added deepfake penalty with weight {args.deepfake_weight}")
        
        # Set up sampler based on whether we're balancing real/fake
        if args.balance_real_fake and hasattr(train, 'get_deepfake_labels'):
            sampler = DeepfakeBalancedBatchSampler(
                train.get_deepfake_labels(), 
                train.get_labels(),
                n_labels=args.n_labels, 
                n_instances=args.n_instances,
                real_ratio=args.real_ratio
            )
            print(f"Using deepfake-balanced sampler with {args.n_labels} labels and {args.n_instances} instances per label")
        else:
            sampler = BalanceSampler(train.get_labels(), n_labels=args.n_labels, n_instances=args.n_instances)
            print(f"Using standard balance sampler with {args.n_labels} labels and {args.n_instances} instances per label")
        
        # Create MixUp or CutMix augmenters if needed
        mixup_augmenter = MixUp(alpha=args.mix_alpha) if args.mixup else None
        cutmix_augmenter = CutMix(alpha=args.mix_alpha) if args.cutmix else None
        
        # Create model with embedding head if needed
        if args.loss in ['arcface', 'cosface', 'adaface']:
            model = ModelWithHead(
                base_model,
                embedding_size=args.embedding_size,
                num_classes=num_classes,
                head_type=args.loss
            ).to(device)
            print(f"Created model with {args.loss} head of size {args.embedding_size}")
        else:
            # For triplet loss, we use the base model directly
            model = base_model
            print("Using base model directly for triplet loss")
            
        # Create optimizer
        optimizer = Adam(model.parameters(), lr=args.lr)
        print(f"Created Adam optimizer with learning rate {args.lr}")
        
        # Log parameters
        params = {
            "model": args.model,
            "loss": args.loss,
            "optimizer": "Adam",
            "learning_rate": args.lr,
            "epochs": epochs,
            "margin": args.margin,
            "scale": args.scale if args.loss in ['arcface', 'cosface', 'adaface'] else None,
            "embedding_size": args.embedding_size if args.loss in ['arcface', 'cosface', 'adaface'] else None,
            "deepfake_weight": args.deepfake_weight,
            "balance_real_fake": args.balance_real_fake,
            "real_ratio": args.real_ratio if args.balance_real_fake else None,
            "miner": args.miner if args.loss == "triplet" else None,
            "advanced_augmentation": args.advanced_augmentation,
            "random_erasing_prob": args.random_erasing_prob if args.advanced_augmentation else None,
            "gaussian_blur_prob": args.gaussian_blur_prob if args.advanced_augmentation else None,
            "jpeg_compression_prob": args.jpeg_compression_prob if args.advanced_augmentation else None,
            "mixup": args.mixup,
            "cutmix": args.cutmix,
            "mix_alpha": args.mix_alpha if (args.mixup or args.cutmix) else None,
            "n_labels": args.n_labels,
            "n_instances": args.n_instances,
            "num_classes": num_classes,
            "seed": args.seed
        }
        mlflow.log_params(params)

        def training():
            epoch_losses = []

            for epoch in range(epochs):
                pbar = tqdm(DataLoader(train, batch_sampler=sampler))
                pbar.set_description(f"epoch: {epoch}/{epochs}")
                batch_losses = []

                for batch in pbar:
                    # Apply MixUp or CutMix if needed
                    if args.mixup and mixup_augmenter is not None:
                        batch = mixup_augmenter(batch)
                    elif args.cutmix and cutmix_augmenter is not None:
                        batch = cutmix_augmenter(batch)
                        
                    embeddings = model(batch["input_tensors"].to(device))
                    
                    # Different loss functions need different inputs
                    if need_labels_directly:
                        # For *Face losses, we need just the class indices
                        loss_kwargs = {}
                        
                        # Add mixed labels if doing mixup/cutmix
                        if (args.mixup or args.cutmix) and 'mixed_labels' in batch and 'lam' in batch:
                            loss_kwargs['mixed_labels'] = batch['mixed_labels'].to(device)
                            loss_kwargs['lam'] = batch['lam']
                            
                        # Add deepfake info if needed
                        if args.deepfake_weight > 0 and "is_deepfake" in batch:
                            loss_kwargs['is_deepfake'] = batch["is_deepfake"].to(device)
                            
                        # Calculate loss
                        loss = criterion(embeddings, batch["labels"].to(device), **loss_kwargs)
                        
                        # Get logs for display
                        if hasattr(criterion, 'last_logs'):
                            postfix = criterion.last_logs
                        else:
                            postfix = {"loss": loss.item()}
                    else:
                        # For Triplet loss, we pass both embeddings and labels
                        loss_kwargs = {}
                        
                        # Add mixed labels if doing mixup/cutmix
                        if (args.mixup or args.cutmix) and 'mixed_labels' in batch and 'lam' in batch:
                            loss_kwargs['mixed_labels'] = batch['mixed_labels'].to(device)
                            loss_kwargs['lam'] = batch['lam']
                            
                        # Add deepfake info if needed
                        if args.deepfake_weight > 0 and "is_deepfake" in batch:
                            loss_kwargs['is_deepfake'] = batch["is_deepfake"].to(device)
                            
                        # Calculate loss
                        loss = criterion(embeddings, batch["labels"].to(device), **loss_kwargs)
                        
                        postfix = criterion.last_logs
                    
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    pbar.set_postfix(postfix)
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
            # For inference, we need to make sure we're getting the base embeddings without the head
            if args.loss in ['arcface', 'cosface', 'adaface']:
                # Create a proper model extractor rather than a function
                embedding_model = EmbeddingExtractor(model).to(device).eval()
                embeddings = inference(embedding_model, val, batch_size=32, num_workers=0, verbose=True)
            else:
                # Set model to eval mode for inference
                model.eval()
                embeddings = inference(model, val, batch_size=32, num_workers=0, verbose=True)
            
            # After inference, set model back to train mode if continuing training
            model.train()
            
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
                    mlflow.log_metric(f"{metric_name}_{k}", value)

        training()
        validation()
        
        # Save model
        model_filename = f"{args.model}_{args.loss}_margin{args.margin}_lr{args.lr}_e{args.epochs}.pth"
        torch.save(model.state_dict(), model_filename)
        mlflow.log_artifact(model_filename)
