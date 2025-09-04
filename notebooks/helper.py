import matplotlib.pyplot as plt
import torch
import wandb
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import math
import random
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import plotly.express as px

# ======================================================
#                 PLOTTING FUNCTIONS
# ======================================================

def plot_images_grid(data, class_names, rows=4, cols=4, predictions=None, figsize=(9, 9)):
    """
    Display a grid of random images from a PyTorch dataset.

    Parameters
    ----------
    data : Dataset
        A torchvision-style dataset that returns (img, label).
    class_names : list
        List containing the class names.
    rows, cols : int
        Number of rows and columns in the grid.
    predictions : list or None
        Optional list of predictions (same indices as the selected images).
    figsize : tuple
        Size of the matplotlib figure.
    """
    fig = plt.figure(figsize=figsize)

    for i in range(1, rows * cols + 1):
        random_idx = torch.randint(0, len(data), size=[1]).item()
        img, label = data[random_idx]

        ax = fig.add_subplot(rows, cols, i)
        ax.imshow(img.squeeze(), cmap="gray")

        if predictions is not None:
            pred_label = class_names[predictions[random_idx]]
            true_label = class_names[label]
            title_color = "green" if pred_label == true_label else "red"
            ax.set_title(f"True: {true_label}\nPredict: {pred_label}", fontsize=8, color=title_color)
        else:
            ax.set_title(class_names[label], fontsize=8)

        ax.axis("off")

    plt.tight_layout()
    plt.show()

    


# ======================================================
#                 DATA LOADING FUNCTIONS
# ======================================================

def create_loader(dataset, batch_size=32, shuffle=True, name="Dataset"):
    """
    Create a DataLoader from a dataset and print basic info.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        PyTorch dataset.
    batch_size : int
        Mini-batch size.
    shuffle : bool
        Whether to shuffle the dataset.
    name : str
        Name of the dataset for printing.

    Returns
    -------
    loader : DataLoader
        PyTorch DataLoader.
    """
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    print(f"{name}: {len(loader)} batches of {batch_size} images")
    images, labels = next(iter(loader))
    print(f"First batch shape - images: {images.shape}, labels: {labels.shape}\n")
    return loader


# ======================================================
#                 METRICS FUNCTIONS
# ======================================================

def accuracy_fn(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Compute the accuracy for a batch of predictions.

    Parameters
    ----------
    logits : torch.Tensor
        Raw outputs from the model (before softmax), shape [batch_size, num_classes].
    labels : torch.Tensor
        True labels, shape [batch_size].

    Returns
    -------
    torch.Tensor
        Accuracy as a float tensor between 0 and 1.
    """
    pred_labels = logits.argmax(dim=1)
    acc = (pred_labels == labels).float().mean()
    return acc


# ======================================================
#                 TRAINING / TESTING FUNCTIONS
# ======================================================

def train_step(model, data_loader, loss_fn, optimizer, accuracy_fn, device):
    """Performs a training loop over one epoch with the model on data_loader"""
    train_loss, train_acc = 0, 0
    model.train()

    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        train_acc += accuracy_fn(logits=y_pred, labels=y).item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train acc: {train_acc*100:.2f}%")
    return train_loss, train_acc


def test_step(model, data_loader, loss_fn, accuracy_fn, device):
    """Performs a testing loop over the model on data_loader"""
    test_loss, test_acc = 0, 0
    model.eval()

    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()
            test_acc += accuracy_fn(logits=y_pred, labels=y).item()

    test_loss /= len(data_loader)
    test_acc /= len(data_loader)
    print(f"Test loss: {test_loss:.5f} | Test acc: {test_acc*100:.2f}%\n")
    return test_loss, test_acc


# ======================================================
#                 W&B TRAINING FUNCTION
# ======================================================

def train_with_wandb(model, train_loader, test_loader, loss_fn, optimizer, accuracy_fn,
                     device, epochs, project_name, hidden_units, dropout=0.0, augmented=False):
    """
    Trains a model and logs metrics to wandb.
    """
    wandb.init(
        project=project_name,
        config = {
            "epochs": epochs,
            "batch_size": train_loader.batch_size,
            "optimizer": optimizer.__class__.__name__,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "model": model.__class__.__name__,
            "hidden_units_list": hidden_units,
            "dropout": dropout,
            "dataset": project_name,
            "data_augmentation": augmented
        }
    )

    for epoch in tqdm(range(epochs)):
        print(f"Epoch: {epoch} \n---------")
        train_loss, train_acc = train_step(model, train_loader, loss_fn, optimizer, accuracy_fn, device)
        test_loss, test_acc = test_step(model, test_loader, loss_fn, accuracy_fn, device)
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc
        })

    wandb.finish()


# ======================================================
#                 W&B DATA FETCHING FUNCTION
# ======================================================

def get_experiments_from_wandb(entity: str, project: str) -> pd.DataFrame:
    """Fetch all experiments from a W&B project and return them as a DataFrame."""
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    data = []

    for run in runs:
        run_info = {
            "name": run.name,
            "epochs": run.config.get("epochs"),
            "batch_size": run.config.get("batch_size"),
            "optimizer": run.config.get("optimizer"),
            "lr": run.config.get("learning_rate"),
            "train_acc": run.summary.get("train_acc"),
            "train_loss": run.summary.get("train_loss"),
            "test_acc": run.summary.get("test_acc"),
            "test_loss": run.summary.get("test_loss"),
            "hidden_units": run.summary.get("hidden_units_list"),
            "url": run.url
        }
        data.append(run_info)

    return pd.DataFrame(data)


# ======================================================
#                 MODEL SAVE / LOAD FUNCTIONS
# ======================================================

def save_model_dict(model, model_name, path):
    """Save a PyTorch model's state_dict to a given path."""
    MODEL_PATH = Path(path)
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    MODEL_SAVE_PATH = MODEL_PATH / model_name
    print(f"Saving model to: {MODEL_SAVE_PATH}")
    torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)
    torch.manual_seed(42)


# ======================================================
#                 PREDICTION FUNCTIONS
# ======================================================

def make_predictions_loader(model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device):
    """Generate predictions from a model given a DataLoader."""
    model.eval()
    all_probs, all_preds = [], []

    with torch.inference_mode():
        for batch in loader:
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())

    return torch.cat(all_probs), torch.cat(all_preds)


def get_error_indices(model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device):
    """Returns the indices of misclassified samples in the dataset."""
    model.eval()
    error_indices = []
    start_idx = 0

    with torch.inference_mode():
        for images, labels in loader:
            batch_size = labels.size(0)
            images, labels = images.to(device), labels.to(device)
            preds = torch.argmax(model(images), dim=1)
            batch_error_idx = (preds != labels).nonzero(as_tuple=True)[0]
            error_indices.extend((batch_error_idx + start_idx).tolist())
            start_idx += batch_size

    return error_indices

# ======================================================
    # Plot missclassified images
# ======================================================


def plot_misclassified_subset(model, dataset, loader, device, num_to_plot=16, cols=4):
    """
    Plots a subset of misclassified images from a dataset.

    Args:
        model: Trained PyTorch model.
        dataset: Dataset (e.g., FashionMNIST_test_data) containing (img, label) tuples.
        loader: DataLoader corresponding to the dataset.
        device: torch.device (CPU or GPU).
        num_to_plot: Number of misclassified images to visualize.
        cols: Number of columns in the grid.

    Returns:
        None. Displays a matplotlib grid of misclassified images.
    """
    
    error_indices = get_error_indices(model, loader, device)
    print(f"Total misclassified images: {len(error_indices)}")

    sample_indices = random.sample(error_indices, min(num_to_plot, len(error_indices)))

    # Create a subset dataset from sampled indices
    error_subset = [dataset[i] for i in sample_indices]

    # Get predictions for the subset
    images_tensor = torch.stack([img for img, _ in error_subset])
    probs, preds = make_predictions_loader(model, [(images_tensor,)], device)

    # Plot the misclassified images using helper function
    class_names = dataset.classes  # list of class names
    rows = math.ceil(len(error_subset) / cols)

    plot_images_grid(error_subset, class_names, rows=rows, cols=cols, predictions=preds.tolist())


# ======================================================
    # Plot confusion matrix
# ======================================================
def plot_confusion_matrix(model, loader, device, class_names, normalize=True, figsize=(8,6)):
    """
    Computes and plots the confusion matrix for a model on a given DataLoader.

    Args:
        model: Trained PyTorch model
        loader: DataLoader for the dataset to evaluate
        device: torch.device
        class_names: list of class names for labels
        normalize: whether to normalize values
        figsize: size of the matplotlib figure
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plot
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

# ======================================================
# Function to plot W&B run metrics
# ======================================================


def plot_wandb_run_metrics(entity: str, project: str, run_id: str):
    """
    Plot train/test accuracy and loss from a W&B run.

    Parameters
    ----------
    entity : str
        W&B username or team name.
    project : str
        W&B project name.
    run_id : str
        Run ID in the project.
    """
    # Login (if not already)
    wandb.login()
    
    #  Access run
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    
    #  Fetch history
    df = run.history()
    if df.empty:
        print("No history found for this run.")
        return
    
    #  Matplotlib: Static plots
   
    plt.figure(figsize=(12,5))

    # Accuracy
    plt.subplot(1,2,1)
    plt.plot(df['epoch'], df['train_acc'], label='Train Accuracy', marker='o')
    plt.plot(df['epoch'], df['test_acc'], label='Test Accuracy', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train vs Test Accuracy')
    plt.legend()
    plt.grid(True)

    # Loss
    plt.subplot(1,2,2)
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o')
    plt.plot(df['epoch'], df['test_loss'], label='Test Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train vs Test Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    #  Plotly: Interactive plots
  
    fig_acc = px.line(df, x='epoch', y=['train_acc','test_acc'], 
                      title='Accuracy per Epoch', labels={'value':'Accuracy','epoch':'Epoch'})
    fig_acc.show()

    fig_loss = px.line(df, x='epoch', y=['train_loss','test_loss'], 
                       title='Loss per Epoch', labels={'value':'Loss','epoch':'Epoch'})
    fig_loss.show()

  
    #  Print links
    print("View run in W&B:", run.url)
    print("View project in W&B:", f"https://wandb.ai/{entity}/{project}")

