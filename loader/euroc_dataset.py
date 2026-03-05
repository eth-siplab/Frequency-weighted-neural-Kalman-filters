import logging
import os
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torchdatasets as td
import yaml
from scipy.interpolate import interp1d


def _read_grayscale_png(path: str) -> np.ndarray:
    if os.path.isfile(path + ".npy"):
        return np.load(path + ".npy")
    data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    np.save(path + ".npy", data)
    return data


class EuRocMAVDataset(td.Dataset):
    SEQUENCE_NAMES = (
        "MH_01_easy",
        "MH_02_easy",
        "MH_03_medium",
        "MH_04_difficult",
        "MH_05_difficult",
    )

    def __init__(
        self,
        root_dir="./data/EuRoC",
        sequence_name="MH_01_easy",
        transform=None,
        download=True,
    ):
        assert sequence_name in self.SEQUENCE_NAMES
        super().__init__()

        self.state_dim = 10
        self.obs_dim = 6
        self.control_dim = 0

        self.root_dir = Path(root_dir)
        self.sequence_name = sequence_name
        self.sequence_dir = self.root_dir / sequence_name / "mav0"
        self.transform = transform

        # Auto-download if data not present
        if download and not self._check_data_exists():
            print(f"Dataset {sequence_name} not found. Downloading...")
            self.download(root_dir, [sequence_name])

        # Load camera data
        cam_file = self.sequence_dir / "cam0/data.csv"
        self.cam_data = pd.read_csv(cam_file)

        # Load IMU data
        imu_file = self.sequence_dir / "imu0/data.csv"
        self.imu_data = pd.read_csv(imu_file)

        # Load ground truth
        gt_file = self.sequence_dir / "state_groundtruth_estimate0/data.csv"
        self.gt_data = pd.read_csv(gt_file)

        # Fix column names by removing any # prefix and trailing spaces
        def _formatting(x: str) -> str:
            return x.replace("#", "").replace("[ns]", "").strip()

        self.cam_data.rename(columns=_formatting, inplace=True)
        self.imu_data.rename(columns=_formatting, inplace=True)
        self.gt_data.rename(columns=_formatting, inplace=True)

        # Camera parameters
        cam_yaml = self.sequence_dir / "cam0/sensor.yaml"
        with open(cam_yaml, "r") as f:
            cam_params = yaml.safe_load(f)
        self.intrinsics = torch.tensor(cam_params["intrinsics"])
        self.distortion = torch.tensor(cam_params["distortion_coefficients"])

        # Create interpolators for ground truth data
        gt_ts = self.gt_data["timestamp"].values
        self.pos_interp = interp1d(
            gt_ts,
            self.gt_data[["p_RS_R_x [m]", "p_RS_R_y [m]", "p_RS_R_z [m]"]].values,
            axis=0,
            fill_value="extrapolate",
        )
        self.quat_interp = interp1d(
            gt_ts,
            self.gt_data[["q_RS_w []", "q_RS_x []", "q_RS_y []", "q_RS_z []"]].values,
            axis=0,
            fill_value="extrapolate",
        )
        self.vel_interp = interp1d(
            gt_ts,
            self.gt_data[
                ["v_RS_R_x [m s^-1]", "v_RS_R_y [m s^-1]", "v_RS_R_z [m s^-1]"]
            ].values,
            axis=0,
            fill_value="extrapolate",
        )

        # Interpolate IMU data for each camera timestamp
        imu_ts = self.imu_data["timestamp"].values
        imu_gyro = self.imu_data[
            ["w_RS_S_x [rad s^-1]", "w_RS_S_y [rad s^-1]", "w_RS_S_z [rad s^-1]"]
        ].values
        imu_accel = self.imu_data[
            ["a_RS_S_x [m s^-2]", "a_RS_S_y [m s^-2]", "a_RS_S_z [m s^-2]"]
        ].values

        self.gyro_interp = interp1d(imu_ts, imu_gyro, axis=0, fill_value="extrapolate")
        self.accel_interp = interp1d(
            imu_ts, imu_accel, axis=0, fill_value="extrapolate"
        )

        logging.getLogger(__name__).info(
            f"EuRocMAVDataset: sequence_name={sequence_name}, state_dim=10, control_dim=0, obs_dim=6."
        )
        cache_path = Path(".cache/data") / f"euroc_{sequence_name}"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache(td.cachers.Pickle(cache_path))

    def _check_data_exists(self):
        """Check if dataset files exist at the expected location"""
        # Check for essential files
        required_files = [
            self.sequence_dir / "cam0/data.csv",
            self.sequence_dir / "cam1/data.csv",
            self.sequence_dir / "imu0/data.csv",
            self.sequence_dir / "state_groundtruth_estimate0/data.csv",
            self.sequence_dir / "cam0/sensor.yaml",
        ]

        # Check if image directory has content
        cam0_dir = self.sequence_dir / "cam0/data"
        cam0_has_images = cam0_dir.exists() and any(cam0_dir.iterdir())

        cam1_dir = self.sequence_dir / "cam1/data"
        cam1_has_images = cam1_dir.exists() and any(cam1_dir.iterdir())

        return (
            all(f.exists() for f in required_files)
            and cam0_has_images
            and cam1_has_images
        )

    def __getitem__(self, idx):
        cam_ts = self.cam_data.iloc[idx]["timestamp"]

        # IMU observation (6)
        gyro = torch.tensor(self.gyro_interp(cam_ts), dtype=torch.float32)
        accel = torch.tensor(self.accel_interp(cam_ts), dtype=torch.float32)
        observations = torch.cat([gyro, accel], dim=-1)

        # State (10): [pos(3), vel(3), quat(4)]
        position = torch.tensor(self.pos_interp(cam_ts), dtype=torch.float32)
        velocity = torch.tensor(self.vel_interp(cam_ts), dtype=torch.float32)
        quaternion = torch.tensor(self.quat_interp(cam_ts), dtype=torch.float32)
        states = torch.cat([position, velocity, quaternion], dim=-1)

        return {
            "states": states.view(1, -1).repeat(
                1, 1
            ),  # keep time axis if you window later
            "observations": observations.view(1, -1).repeat(1, 1),
        }

    @classmethod
    def download(cls, root_dir, sequences=None):
        """Download EuRoc MAV dataset sequences"""
        if sequences is None:
            sequences = cls.SEQUENCE_NAMES

        base_url = (
            "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/"
        )

        root_dir = Path(root_dir)
        root_dir.mkdir(parents=True, exist_ok=True)

        for seq in sequences:
            seq_dir = root_dir / seq
            if seq_dir.exists() and any(seq_dir.iterdir()):
                print(f"{seq} already exists, skipping...")
                continue

            url = f"{base_url}{seq}/{seq}.zip"
            zip_path = root_dir / f"{seq}.zip"

            print(f"Downloading {seq}...")
            urllib.request.urlretrieve(url, zip_path)

            print(f"Extracting {seq}...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(root_dir / seq)

            if zip_path.exists():
                zip_path.unlink()
            print(f"Downloaded and extracted {seq}")


class EuRoCTrackingDataset(torch.utils.data.Dataset):
    """IMU-based state estimation on real EuRoC MAV flight trajectories.

    State  = [px, py, pz, vx, vy, vz, qw, qx, qy, qz]  (10-D, Vicon GT)
    Obs    = [wx, wy, wz, ax, ay, az]                    (6-D, raw IMU)

    The observation model is nonlinear (IMU readings depend on orientation
    and gravity), so models must learn h(x) rather than use a fixed H.

    Machine Hall sequences (MH_01–MH_05) are pooled and windowed into
    overlapping temporal segments.
    """

    SEQUENCE_NAMES = (
        "MH_01_easy",
        "MH_02_easy",
        "MH_03_medium",
        "MH_04_difficult",
        "MH_05_difficult",
        "V1_01_easy",
        "V1_02_medium",
        "V1_03_difficult",
        "V2_01_easy",
        "V2_02_medium",
        "V2_03_difficult",
    )

    # URL patterns per sequence category
    _BASE_URL = "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset"
    _SEQ_CATEGORY = {
        "MH_01_easy": "machine_hall",
        "MH_02_easy": "machine_hall",
        "MH_03_medium": "machine_hall",
        "MH_04_difficult": "machine_hall",
        "MH_05_difficult": "machine_hall",
        "V1_01_easy": "vicon_room1",
        "V1_02_medium": "vicon_room1",
        "V1_03_difficult": "vicon_room1",
        "V2_01_easy": "vicon_room2",
        "V2_02_medium": "vicon_room2",
        "V2_03_difficult": "vicon_room2",
    }

    def __init__(
        self,
        root_dir: str = "./data/EuRoC",
        sequences=None,
        window_size: int = 200,
        stride: int = 50,
        download: bool = True,
        device: str = "cpu",
    ):
        super().__init__()
        self.state_dim = 10  # pos(3) + vel(3) + quat(4)
        self.obs_dim = 6  # gyro(3) + accel(3)
        self.control_dim = 0
        self.device = device

        sequences = sequences or list(self.SEQUENCE_NAMES)
        root_dir = Path(root_dir)

        if download:
            self._download(root_dir, sequences)

        all_obs: list[np.ndarray] = []
        all_states: list[np.ndarray] = []

        for seq in sequences:
            seq_dir = root_dir / seq / "mav0"

            # Load IMU
            imu_file = seq_dir / "imu0" / "data.csv"
            imu = pd.read_csv(imu_file)
            imu.rename(
                columns=lambda c: c.replace("#", "").replace("[ns]", "").strip(),
                inplace=True,
            )
            imu_ts = imu["timestamp"].values
            gyro = imu[
                [
                    "w_RS_S_x [rad s^-1]",
                    "w_RS_S_y [rad s^-1]",
                    "w_RS_S_z [rad s^-1]",
                ]
            ].values
            accel = imu[
                ["a_RS_S_x [m s^-2]", "a_RS_S_y [m s^-2]", "a_RS_S_z [m s^-2]"]
            ].values
            obs = np.concatenate([gyro, accel], axis=1).astype(np.float32)  # [T_imu, 6]

            # Load ground truth
            gt_file = seq_dir / "state_groundtruth_estimate0" / "data.csv"
            gt = pd.read_csv(gt_file)
            gt.rename(
                columns=lambda c: c.replace("#", "").strip(), inplace=True
            )
            gt_ts = gt["timestamp"].values
            pos = gt[["p_RS_R_x [m]", "p_RS_R_y [m]", "p_RS_R_z [m]"]].values
            vel = gt[
                [
                    "v_RS_R_x [m s^-1]",
                    "v_RS_R_y [m s^-1]",
                    "v_RS_R_z [m s^-1]",
                ]
            ].values
            quat = gt[["q_RS_w []", "q_RS_x []", "q_RS_y []", "q_RS_z []"]].values

            # Interpolate GT at IMU timestamps (both ~200Hz but not aligned)
            # Only keep IMU samples within GT time range
            t_min = max(gt_ts[0], imu_ts[0])
            t_max = min(gt_ts[-1], imu_ts[-1])
            mask = (imu_ts >= t_min) & (imu_ts <= t_max)
            imu_ts_valid = imu_ts[mask]
            obs = obs[mask]

            pos_interp = interp1d(gt_ts, pos, axis=0)(imu_ts_valid)
            vel_interp = interp1d(gt_ts, vel, axis=0)(imu_ts_valid)
            quat_interp = interp1d(gt_ts, quat, axis=0)(imu_ts_valid)
            # Renormalize quaternions after interpolation
            quat_norms = np.linalg.norm(quat_interp, axis=1, keepdims=True)
            quat_interp = quat_interp / np.clip(quat_norms, 1e-8, None)

            states = np.concatenate(
                [pos_interp, vel_interp, quat_interp], axis=1
            ).astype(np.float32)  # [T, 10]

            all_obs.append(obs)
            all_states.append(states)

        # Window each sequence into overlapping segments
        obs_windows = []
        state_windows = []
        for obs_seq, state_seq in zip(all_obs, all_states):
            T = obs_seq.shape[0]
            for start in range(0, T - window_size + 1, stride):
                obs_windows.append(obs_seq[start : start + window_size])
                state_windows.append(state_seq[start : start + window_size])

        obs_arr = np.stack(obs_windows)  # [N, W, 6]
        state_arr = np.stack(state_windows)  # [N, W, 10]

        self.observations = torch.tensor(obs_arr, dtype=torch.float32, device=device)
        self.states = torch.tensor(state_arr, dtype=torch.float32, device=device)

        logging.getLogger(__name__).info(
            f"EuRoCTrackingDataset: {len(self)} windows "
            f"(seqs={sequences}, W={window_size}, stride={stride})"
        )

    @classmethod
    def _download(cls, root_dir: Path, sequences):
        """Download EuRoC sequences, extracting only IMU + ground truth."""
        root_dir = Path(root_dir)
        root_dir.mkdir(parents=True, exist_ok=True)

        for seq in sequences:
            imu_csv = root_dir / seq / "mav0" / "imu0" / "data.csv"
            gt_csv = (
                root_dir / seq / "mav0" / "state_groundtruth_estimate0" / "data.csv"
            )
            if imu_csv.exists() and gt_csv.exists():
                continue

            cat = cls._SEQ_CATEGORY.get(seq)
            if cat is None:
                raise ValueError(f"Unknown sequence {seq}")
            url = f"{cls._BASE_URL}/{cat}/{seq}/{seq}.zip"
            zip_path = root_dir / f"{seq}.zip"

            print(f"Downloading {seq} from {url} ...")
            urllib.request.urlretrieve(url, zip_path)

            # Extract only imu0/ and state_groundtruth_estimate0/
            print(f"Extracting IMU + GT from {seq}...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                for member in zf.namelist():
                    if "imu0/" in member or "state_groundtruth_estimate0/" in member:
                        zf.extract(member, root_dir / seq)

            if zip_path.exists():
                zip_path.unlink()
            print(f"Done: {seq}")

    def __len__(self) -> int:
        return self.states.shape[0]

    def __getitem__(self, idx: int):
        return {
            "states": self.states[idx],  # [W, 10]
            "observations": self.observations[idx],  # [W, 6]
        }
