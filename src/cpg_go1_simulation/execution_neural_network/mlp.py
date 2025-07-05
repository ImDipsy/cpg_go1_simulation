import pathlib

import numpy as np
import pandas
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from cpg_go1_simulation.config import TRAIN_DATA_DIR, TRAIN_RESULT_DIR

# Version number
VERSION = "1.2"


class CustomDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).reshape(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MLP(nn.Module):
    def __init__(self, output_size=1):
        super(MLP, self).__init__()

        # Define the dimensions of the input features
        self.onehot1_dim = 3
        self.onehot2_dim = 5
        self.continuous_dim = 3
        self.total_input_dim = self.onehot1_dim + self.onehot2_dim + self.continuous_dim

        # Define the layers
        # Layer 1: Input -> 24
        self.layer1 = nn.Sequential(
            nn.Linear(self.total_input_dim, 24),
            nn.ReLU(),
        )

        # Layer 2: First hidden layer -> Second hidden layer
        self.layer2 = nn.Sequential(nn.Linear(24, 12), nn.ReLU())

        # Output layer: Second hidden layer -> Output
        self.output_layer = nn.Linear(12, output_size)

    def preprocess_input(self, x):
        """
        Preprocess the input data, without normalization:
        - 0-2 columns: First one-hot encoding
        - 3-7 columns: Second encoding
        - 8-10 columns: Continuous variables (used as is)
        """
        # Split the input data
        onehot1_features = x[:, : self.onehot1_dim]
        onehot2_features = x[:, self.onehot1_dim : self.onehot1_dim + self.onehot2_dim]
        # Use original values directly
        continuous_features = x[:, -self.continuous_dim :]

        # Combine all features
        return torch.cat(
            [onehot1_features, onehot2_features, continuous_features], dim=1
        )

    def forward(self, x):
        x = self.preprocess_input(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.output_layer(x)
        return x


def process_data(file_path: str):
    """
    Process the raw data files and save the processed data

    data format:
    - column 1: gait type
    - column 2: stance length ratio
    - column 3: neuron_X
    - column 4: derivative of neuron_X/50
    - column 5: joint position calculated by inverse kinematics
    """

    # One-hot encoding sequence 1: Joint number i (1-8) corresponds to onehot1[i-1]
    onehot1 = [
        [0, 1, 0],
        [0, 1, 1],
        [0, 0, 1],
        [0, 0, 0],
        [1, 1, 0],
        [1, 1, 1],
        [1, 0, 1],
        [1, 0, 0],
    ]

    # One-hot encoding sequence 2: Gait type
    gait_to_onehot = {
        0.2: [1, 0, 0, 0, 0],
        0.4: [0, 1, 0, 0, 0],
        0.6: [0, 0, 1, 0, 0],
        0.8: [0, 0, 0, 1, 0],
        1.0: [0, 0, 0, 0, 1],
    }
    # Read all files
    files = sorted(list(file_path.glob("neuron_*.csv")))

    # store processed data
    processed_data = []

    # process each file
    for i, file in enumerate(files):
        # Read the data
        df = pandas.read_csv(file, header=None)
        data = df.to_numpy()

        # Get the encoding for the joint number
        current_onehot1 = onehot1[i]

        # Process each row
        for row in data:
            # Get the one-hot encoding for the gait type
            gait_type = row[0]  # first column is the gait type
            current_onehot2 = gait_to_onehot[gait_type]

            # combine the one-hot encodings and the original data
            new_row = np.concatenate(
                [
                    current_onehot1,  # joint number's three-bit encoding (3-bit)
                    current_onehot2,  # gait type's five-bit encoding (5-bit)
                    row[1:5],  # original data (4-bit)
                ]
            )

            processed_data.append(new_row)

    processed_data = np.array(processed_data)
    df_processed = pandas.DataFrame(processed_data)

    # save the processed data
    save_path = file_path / "processed_data.csv"
    df_processed.to_csv(save_path, index=False, header=False)

    print(f"Data has been processed,  {len(processed_data)} Rows Data")
    print(f"Data shape is: {processed_data.shape}")
    print(f"Data has been saved at: {save_path}")

    return processed_data


def load_and_prepare_data(file_path, test_size=0.2, random_state=42):
    """
    Load the processed data and split it into training and validation sets
    """
    # processed_data = process_data(file_path)
    df = pandas.read_csv(file_path, header=None)
    processed_data = df.to_numpy()

    # Split the data into features and labels
    X = processed_data[:, :-1]  # All columns except the last one
    y = processed_data[:, -1]  # The last column is the label

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_val, y_train, y_val


def evaluate_model(model, data_loader, device):
    """
    Evaluate the model on the given data and return the metrics
    """
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(y.cpu().numpy())

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Calculate the metrics
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - actuals))

    # Calculate R2 score
    ss_res = np.sum((actuals - predictions) ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}


def train_model(
    model,
    root_dir,
    train_loader,
    val_loader,
    num_epochs=100,
    learning_rate=0.01,
    device="cuda",
):
    """
    Train the model and return the trained model and training history
    """
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    best_val_loss = float("inf")
    training_history = {"train_loss": [], "val_loss": [], "learning_rate": []}

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                val_loss += criterion(outputs, batch_y).item()

        # Calculate the average loss for this epoch
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        # Update the learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # Record the loss and learning rate
        training_history["train_loss"].append(train_loss)
        training_history["val_loss"].append(val_loss)
        training_history["learning_rate"].append(current_lr)

        # Save the model if the validation loss is the best so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = root_dir / "DATA/Training/train_model"
            model_path.mkdir(exist_ok=True)

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "model_config": {
                    "input_dim": model.total_input_dim,
                    "output_size": model.output_layer.out_features,
                    "version": VERSION,  # Use version number
                    "normalize": False,
                },
                "training_config": {
                    "learning_rate": learning_rate,
                    "num_epochs": num_epochs,
                    "device": str(device),
                    "batch_size": train_loader.batch_size,
                },
            }

            try:
                torch.save(
                    checkpoint,
                    model_path / f"best_model_v{VERSION}.pt",  # Version number
                    _use_new_zipfile_serialization=False,
                )
                print(
                    f"Module has been saved at: {model_path}/best_model_v{VERSION}.pt"
                )
            except Exception as e:
                print(f"Saved Module Error: {e}")

        # Print the loss and learning rate for this epoch
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], "
                f"Train Loss: {train_loss:.6f}, "
                f"Val Loss: {val_loss:.6f}, "
                f"Learning Rate: {current_lr:.6f}"
            )

        if val_loss < 1e-4 or learning_rate < 1e-5:
            break

    return model, training_history


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and prepare the data
    root_dir = pathlib.Path(__file__).parent.parent
    file_path = TRAIN_DATA_DIR / "training_data.csv"
    X_train, X_val, y_train, y_val = load_and_prepare_data(file_path)

    # Create data loaders
    train_dataset = CustomDataset(X_train, y_train)
    val_dataset = CustomDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Create the model
    model = MLP()

    # Train the model
    model, history = train_model(
        model, root_dir, train_loader, val_loader, device=device
    )

    # Evaluate the model
    print("\n: Training Set Evaluation:")
    train_metrics = evaluate_model(model, train_loader, device)
    for metric, value in train_metrics.items():
        print(f"{metric}: {value:.6f}")

    print("\n Validation Set Evaluation:")
    val_metrics = evaluate_model(model, val_loader, device)
    for metric, value in val_metrics.items():
        print(f"{metric}: {value:.6f}")

    # visualize the training history
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.plot(history["train_loss"], label="Train Loss")
        plt.plot(history["val_loss"], label="Validation Loss")
        plt.title("Training and Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history["learning_rate"])
        plt.title("Learning Rate")
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")

        plt.tight_layout()
        result_path = TRAIN_RESULT_DIR
        result_path.mkdir(exist_ok=True)
        plt.savefig(result_path / f"training_history_v{VERSION}.png")
        print(f"\nTraining history has been saved at 'training_history_v{VERSION}.png'")
    except ImportError:
        print("matplotlib not installed, skipping visualization")

    print("\n Training Completed!")
    print(f"Version: {VERSION}")


if __name__ == "__main__":
    main()
