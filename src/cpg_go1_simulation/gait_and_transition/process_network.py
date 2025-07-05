import logging

import numpy as np
import torch

from cpg_go1_simulation.config import BEST_MODEL_PATH
from cpg_go1_simulation.execution_neural_network.mlp import MLP


class ProcessNetwork:
    def __init__(self):
        # load the model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _ = self._load_model()

    def _load_model(self):
        """Load the trained model"""
        model_path = BEST_MODEL_PATH

        try:
            # Check if the file exists
            if not model_path.exists():
                raise FileNotFoundError(f"Model file does not exist: {model_path}")

            # Load the saved data
            logging.info(f"Loading model: {model_path}")
            checkpoint = torch.load(
                model_path, map_location=self.device, weights_only=False
            )

            # Get version number
            version = checkpoint.get("model_config", {}).get("version", "1.2")
            logging.info(f"Model version: v{version}")

            # Create model
            model = MLP()

            # Validate checkpoint format
            required_keys = [
                "model_state_dict",
                "model_config",
                "training_config",
            ]
            for key in required_keys:
                if key not in checkpoint:
                    raise KeyError(f"Checkpoint is missing required key: {key}")

            # Load model state
            try:
                model.load_state_dict(checkpoint["model_state_dict"], strict=False)
                logging.info("Model state loaded successfully")
            except Exception as e:
                logging.warning(f"Warning occurred while loading model state: {e}")

            # Log model configuration information
            if "model_config" in checkpoint:
                logging.info(f"Model configuration: {checkpoint['model_config']}")
            if "training_config" in checkpoint:
                logging.info(f"Training configuration: {checkpoint['training_config']}")
            if "best_val_loss" in checkpoint:
                logging.info(f"Best validation loss: {checkpoint['best_val_loss']:.6f}")
            if "epoch" in checkpoint:
                logging.info(f"Training epoch: {checkpoint['epoch'] + 1}")

            model = model.to(self.device)
            model.eval()
            logging.info("Model loaded successfully")
            return model, version

        except Exception as e:
            logging.error(f"Error occurred while loading model: {e}")
            raise e

    def _process_data(self, cpg_data, gait_type):
        """Process the CPG data to the required format"""
        processed_data = []
        onehot_gait = self._get_gait_onehot(gait_type)

        for i in range(8):
            features = np.concatenate(
                [
                    self._get_joint_onehot(i),
                    onehot_gait,
                    [0.8],
                    [cpg_data[i]],
                    [cpg_data[i + 8] * 1 / 50],
                ]
            )
            processed_data.append(features)

        return processed_data

    def _get_joint_onehot(self, joint_idx):
        """Get encoding for the joint index"""
        encodings = [
            [0, 1, 0],
            [0, 1, 1],
            [0, 0, 1],
            [0, 0, 0],
            [1, 1, 0],
            [1, 1, 1],
            [1, 0, 1],
            [1, 0, 0],
        ]
        return encodings[joint_idx]

    def _get_gait_onehot(self, gait_type):
        """Get encoding for the gait type"""
        gait_map = {
            "walk": [1, 0, 0, 0, 0],
            "trot": [0, 1, 0, 0, 0],
            "pace": [0, 0, 1, 0, 0],
            "bound": [0, 0, 0, 1, 0],
            "pronk": [0, 0, 0, 0, 1],
        }
        return gait_map[gait_type]

    def _predict_joint_angles(self, processed_data):
        """Predict joint angles from processed data"""
        processed_data = np.array(processed_data)
        with torch.no_grad():
            input_tensor = torch.FloatTensor(processed_data).to(self.device)
            predictions = self.model(input_tensor).cpu().numpy()

        joint_angles = np.zeros(12)

        mapping = {
            0: 1,
            1: 4,
            2: 7,
            3: 10,
            4: 2,
            5: 5,
            6: 8,
            7: 11,
        }

        predictions = predictions.reshape(-1)
        for pred_idx, joint_idx in mapping.items():
            joint_angles[joint_idx] = predictions[pred_idx]

        return joint_angles
