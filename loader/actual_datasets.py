# actual_dataset_loader.py
import warnings
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset

from .euroc_dataset import EuRocMAVDataset
from .nclt_dataset import NCLTDataset
from .robotcar_dataset import OxfordRobotCarDataset
from .uipdp_dataset import UWBIMUDataset

warnings.filterwarnings("ignore")


class UWBIMUWrapper(Dataset):
    """
    UWB-IMU Dataset wrapper for Kalman filter training.
    Returns states, observations, and controls in standard format.
    """

    def __init__(
        self,
        root_dir: str,
        session_name: str = "00_session2_1",
        input_type: str = "raw",
        label_type: str = "ekf",
        uwb_repr: str = "full",
        train: bool = True,
        device: str = "cpu",
    ):
        self.device = device

        # Initialize the actual dataset
        self.dataset = UWBIMUDataset(
            root_dir=root_dir,
            session_name=session_name,
            input_type=input_type,
            label_type=label_type,
            uwb_repr=uwb_repr,
            train=train,
        )

        # State and observation dimensions
        self.state_dim = 18  # 6 devices × 3 dimensions each
        self.obs_dim = 18  # 6 devices × 3 dimensions each
        self.control_dim = 12  # Controls

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.dataset[idx]

        # Extract IMU and UWB data
        imu_target = sample["imu_target"]  # (seq_len, 72)
        uwb_target = sample["uwb_target"]  # (seq_len, 36) or (seq_len, 15)
        imu_input = sample["imu_input"]  # (seq_len, 72)
        uwb_input = sample["uwb_input"]  # (seq_len, 36) or (seq_len, 15)
        imu_control = sample["imu_control"]  # (seq_len, 72)

        # Combine IMU and UWB for states and observations
        # Take first 18 dimensions from IMU (acceleration data)
        states = torch.cat(
            [
                imu_target[:, :18],  # IMU acceleration ground truth
            ],
            dim=-1,
        )

        observations = torch.cat(
            [
                imu_input[:, :18],  # IMU acceleration observations
            ],
            dim=-1,
        )

        controls = imu_control[:, :12]  # First 12 control dimensions

        return {
            "states": states.to(self.device),
            "observations": observations.to(self.device),
            "controls": controls.to(self.device),
        }


class EuRoCWrapper(Dataset):
    """
    EuRoC MAV Dataset wrapper for Kalman filter training.
    Returns states, observations, and controls in standard format.
    """

    def __init__(
        self,
        root_dir: str,
        sequence_name: str = "MH_01_easy",
        download: bool = True,
        device: str = "cpu",
    ):
        self.device = device

        # Initialize the actual dataset
        self.dataset = EuRocMAVDataset(
            root_dir=root_dir,
            sequence_name=sequence_name,
            download=download,
        )

        # State and observation dimensions
        self.state_dim = 10  # position(3) + velocity(3) + orientation(4)
        self.obs_dim = 6  # IMU data (gyro + accel)
        self.control_dim = 0  # No control inputs

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.dataset[idx]

        # Extract data
        position = sample["position"]  # (3,)
        velocity = sample["velocity"]  # (3,)
        orientation = sample["orientation"]  # (4,)
        imu_data = sample["imu"]  # (6,)

        # Combine into state vector
        states = torch.cat([position, velocity, orientation], dim=-1)
        observations = imu_data
        controls = torch.zeros(0)  # No controls

        return {
            "states": states.to(self.device),
            "observations": observations.to(self.device),
            "controls": controls.to(self.device),
        }


class NCLTWrapper(Dataset):
    """
    NCLT Dataset wrapper for Kalman filter training.
    Returns states, observations, and controls in standard format.
    """

    def __init__(
        self,
        root_dir: str,
        date: str = "2012-01-08",
        download: bool = True,
        device: str = "cpu",
    ):
        self.device = device

        # Initialize the actual dataset
        self.dataset = NCLTDataset(
            root_dir=root_dir,
            date=date,
            download=download,
        )

        # State and observation dimensions
        self.state_dim = 3  # [x, y, theta] pose
        self.obs_dim = 6  # odometry data
        self.control_dim = 6  # odometry differences

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.dataset[idx]

        # Extract data
        true_poses = sample["true_poses"]  # (3,) - [x, y, theta]
        odom = sample["odom"]  # (6,) - odometry
        odom_diff = sample["odom_diff"]  # (6,) - control inputs

        return {
            "states": true_poses.to(self.device),
            "observations": odom.to(self.device),
            "controls": odom_diff.to(self.device),
        }


class RobotCarWrapper(Dataset):
    """
    Oxford RobotCar Dataset wrapper for Kalman filter training.
    Returns states, observations, and controls in standard format.
    """

    def __init__(
        self,
        root_dir: str,
        traversal: str = "2014-05-19-13-20-57",
        sensors: list = None,
        use_radar: bool = False,
        download: bool = False,
        device: str = "cpu",
    ):
        self.device = device

        if sensors is None:
            sensors = ["gps_ins", "lms_front"]

        # Initialize the actual dataset
        self.dataset = OxfordRobotCarDataset(
            root_dir=root_dir,
            traversal=traversal,
            sensors=sensors,
            use_radar=use_radar,
            download=download,
        )

        # State and observation dimensions
        self.state_dim = 6  # position(3) + orientation(3)
        self.obs_dim = 10  # sensor observations (simplified)
        self.control_dim = 0  # No control inputs

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.dataset[idx]

        # Extract pose data if available
        if "position" in sample and "orientation" in sample:
            position = sample["position"]  # (3,)
            orientation = sample["orientation"]  # (3,)
            states = torch.cat([position, orientation], dim=-1)
        else:
            # Fallback to dummy state
            states = torch.zeros(6)

        # Create observations from available sensor data
        observations = []

        # Add LiDAR data if available
        lidar_keys = [k for k in sample.keys() if "points" in k]
        if lidar_keys:
            lidar_data = sample[lidar_keys[0]]  # Take first LiDAR
            # Simplify to mean coordinates
            if lidar_data.numel() > 0:
                mean_coords = lidar_data.mean(dim=0)[:3]  # [x_mean, y_mean, z_mean]
                observations.append(mean_coords)
            else:
                observations.append(torch.zeros(3))

        # Add radar data if available
        if "radar_scan" in sample:
            radar_data = sample["radar_scan"]
            # Simplify radar to summary statistics
            radar_summary = torch.tensor(
                [
                    radar_data.mean(),
                    radar_data.std(),
                    radar_data.max(),
                ]
            )
            observations.append(radar_summary)

        # Pad observations to fixed size
        if observations:
            obs = torch.cat(observations)
            if obs.shape[0] < self.obs_dim:
                padding = torch.zeros(self.obs_dim - obs.shape[0])
                obs = torch.cat([obs, padding])
            else:
                obs = obs[: self.obs_dim]
        else:
            obs = torch.zeros(self.obs_dim)

        controls = torch.zeros(0)  # No controls

        return {
            "states": states.to(self.device),
            "observations": obs.to(self.device),
            "controls": controls.to(self.device),
        }


# Dataset registry
ACTUAL_DATASETS = {
    "uwbimu": UWBIMUWrapper,
    "euroc": EuRoCWrapper,
    "nclt": NCLTWrapper,
    "robotcar": RobotCarWrapper,
}


def create_actual_dataset(dataset_name: str, **kwargs) -> Dataset:
    """
    Create an actual dataset by name.

    Args:
        dataset_name: Name of the dataset ("uwbimu", "euroc", "nclt", "robotcar")
        **kwargs: Dataset-specific arguments

    Returns:
        Dataset instance
    """
    if dataset_name not in ACTUAL_DATASETS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Available: {list(ACTUAL_DATASETS.keys())}"
        )

    return ACTUAL_DATASETS[dataset_name](**kwargs)


if __name__ == "__main__":
    """
    Test actual datasets
    """
    print("Testing Actual Datasets")
    print("=" * 40)

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Test UWB-IMU dataset (if available)
    try:
        print("Testing UWB-IMU Dataset...")
        uwbimu_dataset = UWBIMUWrapper(
            root_dir="/path/to/uwbimu/data",
            session_name="00_session2_1",
            train=True,
        )
        print(f"  Dataset length: {len(uwbimu_dataset)}")
        sample = uwbimu_dataset[0]
        print(f"  States shape: {sample['states'].shape}")
        print(f"  Observations shape: {sample['observations'].shape}")
        print(f"  Controls shape: {sample['controls'].shape}")
    except Exception as e:
        print(f"  UWB-IMU dataset not available: {e}")

    # Test EuRoC dataset (if available)
    try:
        print("\nTesting EuRoC Dataset...")
        euroc_dataset = EuRoCWrapper(
            root_dir="/path/to/euroc/data",
            sequence_name="MH_01_easy",
            download=False,
        )
        print(f"  Dataset length: {len(euroc_dataset)}")
        sample = euroc_dataset[0]
        print(f"  States shape: {sample['states'].shape}")
        print(f"  Observations shape: {sample['observations'].shape}")
        print(f"  Controls shape: {sample['controls'].shape}")
    except Exception as e:
        print(f"  EuRoC dataset not available: {e}")

    print("\nActual datasets ready for Kalman filter training!")
