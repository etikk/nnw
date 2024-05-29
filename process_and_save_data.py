#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import torch
from transformers import BertForMaskedLM
import numpy as np
from scipy.stats import entropy
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image, ImageDraw
from multiprocessing import Pool
from tqdm import tqdm
import matplotlib.pyplot as plt
import traceback

# BLOCK 5
# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
print("Tokenizer initialized")

def preprocess_and_split_data(dataset, min_tokens=256, max_length=256):
    #print("Called preprocess_and_split_data")

    """
    Tokenize, filter, and split the dataset.
    - Tokenize texts
    - Filter out entries with fewer than min_tokens
    - Cap the token length at max_length
    - Split the dataset into train, validation, and test sets
    """
    # Tokenize texts
    dataset['tokens'] = dataset['text'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True, truncation=True, max_length=max_length))

    # Filter out rows where the token length is less than min_tokens
    dataset = dataset[dataset['tokens'].apply(len) >= min_tokens]

    # Split the dataset into train, validation, and test sets
    train_data, temp_data = train_test_split(dataset, test_size=0.3, random_state=42)  # Splitting 70% for train, 30% for temp_data
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)  # Splitting temp_data into 50% validation and 50% test

    return train_data, val_data, test_data

def verify_token_lengths_and_labels(data, label='Train'):
    #print("Called verify_token_lengths_and_labels")

    print(f"--- {label} Data Verification ---")
    print(f"{label} set size: {len(data)}")
    all_256_tokens = all(len(tokens) == 256 for tokens in data['tokens'])
    print(f"All data points are 256 tokens: {all_256_tokens}")
    if all_256_tokens:
        print(f"Sample labels from {label} data:")
        print(data['label'].sample(5).to_list())

# BLOCK 11
# Initialize the model
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()
print("BERT model initialized")

def calculate_metrics(token_ids):
    #print("Called calculate_metrics")

    """Calculate metrics for a single text represented by token IDs."""
    # Convert token IDs to tensor
    input_ids = torch.tensor([token_ids])

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    probabilities = torch.softmax(logits, dim=-1)
    token_probabilities = probabilities[0, :-1].max(dim=-1).values.numpy()
    token_entropy = entropy(probabilities[0, :-1].numpy(), axis=1)
    token_embeddings = model.bert.embeddings.word_embeddings(input_ids)[0].detach().numpy()

    # Calculate semantic similarity (cosine similarity with the average embedding)
    avg_embedding = np.mean(token_embeddings, axis=0)
    similarity_scores = cosine_similarity(token_embeddings, avg_embedding.reshape(1, -1)).flatten()

    # Token lengths
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    token_lengths = np.array([len(token) for token in tokens])

    # Filter out special tokens ([CLS] and [SEP])
    valid_indices = [i for i, token in enumerate(tokens) if token not in ["[CLS]", "[SEP]"]]
    tokens = [tokens[i] for i in valid_indices]
    token_probabilities = token_probabilities[valid_indices]
    token_entropy = token_entropy[valid_indices]
    similarity_scores = similarity_scores[valid_indices]
    token_lengths = token_lengths[valid_indices]

    # Normalize metrics to 0-255
    token_probabilities_normalized = (token_probabilities / token_probabilities.max() * 255).astype(np.uint8)
    token_entropy_normalized = (token_entropy / token_entropy.max() * 255).astype(np.uint8)
    similarity_scores_normalized = (similarity_scores / similarity_scores.max() * 255).astype(np.uint8)
    # token_lengths_normalized = (token_lengths / token_lengths.max() * 255).astype(np.uint8)

    return {
        'tokens': tokens,
        'probabilities': token_probabilities_normalized,
        'entropy': token_entropy_normalized,
        'similarity': similarity_scores_normalized,
        'lengths': token_lengths
    }

def pad_or_truncate_metrics(metrics, target_length=256):
    #print("Called pad_or_truncate_metrics")

    """Pad or truncate the metrics to ensure the token array has the target length."""
    current_length = len(metrics['tokens'])
    if current_length < target_length:
        for key in metrics.keys():
            padding = [0] * (target_length - current_length)
            metrics[key] = np.concatenate((metrics[key], padding))
    elif current_length > target_length:
        for key in metrics.keys():
            metrics[key] = metrics[key][:target_length]
    return metrics

def spread_normalization(data, scale=255):
    #print("Called spread_normalization")

    """Normalize data using a square root transformation to reduce high-end compression."""
    sqrt_data = np.sqrt(data)
    normalized_data = (sqrt_data - np.min(sqrt_data)) / (np.max(sqrt_data) - np.min(sqrt_data))
    return (normalized_data * scale).astype(int)

def clip_and_normalize_metrics(metrics):
    #print("Called clip_and_normalize_metrics")

    """Adjusts the metrics by clipping to a range between the lowest non-zero and highest non-255 values for 'probabilities' and 'similarity'."""
    keys_to_normalize = ['probabilities', 'similarity']
    for key in keys_to_normalize:
        if key in metrics:
            data = metrics[key]
            min_val = np.min(data[data > 0])
            max_val = np.max(data[data < 255])
            clipped_data = np.where(data == 0, min_val, data)
            clipped_data = np.where(clipped_data == 255, max_val, clipped_data)
            metrics[key] = ((clipped_data - min_val) / (max_val - min_val) * 255).astype(int)

    metrics['probabilities'] = np.where((255 - metrics['probabilities']) * 10 < 255,
        metrics['probabilities'] - ((255 - metrics['probabilities']) * 10),
        metrics['probabilities'])
    return metrics

def generate_circle_image(metrics, img_width=256, img_height=256):
    #print("Called generate_circle_image")

    """Generate a 256x256 pixel grayscale image with circles based on metrics."""
    img = Image.new('L', (img_width, img_height), color='white')  # 'L' mode for grayscale
    draw = ImageDraw.Draw(img)

    centers = []  # To store center points for the polyline

    for idx in range(len(metrics['tokens'])):
        # Normalize coordinates to fit within the image dimensions
        x_center = int(metrics['probabilities'][idx])
        y_center = int(metrics['similarity'][idx])
        centers.append((x_center, y_center))

        # radius = 2
        radius = int(metrics['lengths'][idx])
        entropy = int(metrics['entropy'][idx])  # Grayscale intensity from 0 to 255

        # Calculate top left and bottom right coordinates of the circle
        top_left = (x_center - radius, y_center - radius)
        bottom_right = (x_center + radius, y_center + radius)

        # Draw the circle
        draw.ellipse([top_left, bottom_right], outline=entropy)

    # Draw the polyline connecting all centers
    draw.line(centers, fill='black', width=1)

    return img

def process_single_entry(entry):
    #print("Called process_single_entry")

    token_ids, label = entry
    try:
        metrics = calculate_metrics(token_ids)
        metrics = pad_or_truncate_metrics(metrics)
        all_metrics = metrics  # Store original metrics before further manipulation

        # Apply normalization and adjustments for image generation
        metrics = clip_and_normalize_metrics(metrics)
        img = generate_circle_image(metrics)
        return np.array(img), all_metrics, label
    except Exception as e:
        print(f"Error processing entry: {entry}")
        print(traceback.format_exc())
        return None, None, None

#def process_images_and_metrics(tokens, labels):
    print("Called process_images_and_metrics")

    images = []
    all_metrics = []
    all_labels = []
    with Pool(processes=2) as pool:  # Adjust number of processes based on available CPU cores
        results = list(tqdm(pool.imap(process_single_entry, zip(tokens, labels)), total=len(tokens)))
    for result in results:
        img, metrics, label = result
        if img is not None:
            images.append(img)
            all_metrics.append(metrics)
            all_labels.append(label)
    return images, all_metrics, all_labels

def process_images_and_metrics(tokens, labels):
    print("Called process_images_and_metrics")

    images = []
    all_metrics = []
    all_labels = []
    for entry in tqdm(zip(tokens, labels), total=len(tokens)):
        img, metrics, label = process_single_entry(entry)
        if img is not None:
            images.append(img)
            all_metrics.append(metrics)
            all_labels.append(label)
    return images, all_metrics, all_labels

# Sanity check for image and mterics generation
import matplotlib.pyplot as plt

def plot_image(image, title):
    """Helper function to plot an image with a title."""
    plt.figure(figsize=(2, 2))
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def print_metrics(metrics, idx):
    """Helper function to print selected metrics."""
    print(f"Metrics for Entry {idx}:")
    keys_to_display = ['probabilities', 'entropy', 'similarity', 'lengths']  # Example keys
    for key in keys_to_display:
        if key in metrics:
            print(f"{key}: {metrics[key][:5]}")  # Print first 5 values as an example

def sanity_check_images_and_metrics(images, metrics, step=400):
    """Performs a sanity check by displaying images and printing metrics at specified intervals."""
    for idx in range(0, len(images), step):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(2.56, 2.56))  # 256x256 pixels
        plt.imshow(images[idx])
        plt.title(f"Sample Image at Index {idx}")
        plt.axis('off')
        plt.show()
        plot_image(images[idx], f"Sample Image at Index {idx}")
        print_metrics(metrics[idx], idx)

def main():
    print("Main function started")

    # Read the dataset
    dataset_train = pd.read_csv('final_train.csv')
    dataset_test = pd.read_csv('final_test.csv')
    print("Datasets loaded")

    # Limit dataset size for pipeline testing
    # Comment this out in real pipeline
    dataset_train = dataset_train[:100]
    dataset_test = dataset_test[:100]
    print("Datasets limited to 100 entries each")

    # Combine train and test datasets for preprocessing
    combined_dataset = pd.concat([dataset_train, dataset_test])
    print("Datasets combined")

    # Preprocess and split the dataset
    train_data, val_data, test_data = preprocess_and_split_data(combined_dataset)
    print("Datasets preprocessed and split")

    print(f"Training set size: {len(train_data)}")
    print(f"Validation set size: {len(val_data)}")
    print(f"Test set size: {len(test_data)}")

    # Process and save data with labels
    print("Processing and saving data")
    train_images, train_metrics, train_labels = process_images_and_metrics(train_data['tokens'], train_data['label'])
    val_images, val_metrics, val_labels = process_images_and_metrics(val_data['tokens'], val_data['label'])
    test_images, test_metrics, test_labels = process_images_and_metrics(test_data['tokens'], test_data['label'])

    # Save to .npz files
    print("Saving data to .npz files")
    np.savez_compressed('npz_results/train_data.npz', images=train_images, metrics=train_metrics, labels=train_labels)
    np.savez_compressed('npz_results/val_data.npz', images=val_images, metrics=val_metrics, labels=val_labels)
    np.savez_compressed('npz_results/test_data.npz', images=test_images, metrics=test_metrics, labels=test_labels)


    # Sanity check for training data
    sanity_check_images_and_metrics(train_images, train_metrics)
    sanity_check_images_and_metrics(val_images, val_metrics)
    sanity_check_images_and_metrics(test_images, test_metrics)


if __name__ == "__main__":
    main()