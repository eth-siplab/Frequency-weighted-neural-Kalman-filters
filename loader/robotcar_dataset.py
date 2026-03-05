import json
import logging
import os
import tarfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torchdatasets as td
from tqdm import tqdm


class OxfordRobotCarDataset(td.Dataset):
    """
    Oxford RobotCar Dataset loader.

    This dataset loader handles the Oxford RobotCar Dataset, which includes data from various
    sensors such as stereo cameras, LiDAR, GPS/INS, and optionally radar.

    References:
    - Original dataset: "1 Year, 1000km: The Oxford RobotCar Dataset"
      https://robotcar-dataset.robots.ox.ac.uk/
    - Radar extension: "The Oxford Radar RobotCar Dataset"
      http://ori.ox.ac.uk/datasets/radar-robotcar-dataset/
    """

    # Define available traversals from the original dataset
    TRAVERSALS = {
        "2014-05-19-13-20-57": "standard",  # Example traversal
        "2014-12-16-09-14-09": "rain",  # Rain example
        "2015-02-03-08-45-10": "snow",  # Snow example
    }

    # Define sensors available in the dataset
    SENSORS = {
        "stereo": ["left", "center", "right"],
        "mono": ["left", "right", "rear"],
        "lidar": ["lms_front", "lms_rear", "ldmrs", "velodyne_left", "velodyne_right"],
        "radar": ["navtech"],
        "gps": ["ins"],
    }

    def __init__(
        self,
        root_dir: str = "./data/oxford",
        # traversal: str = "2014-05-19-13-20-57",
        traversal: str = "sample",
        sensors: List[str] = None,
        use_radar: bool = False,
        transform=None,
        download: bool = False,
        cache_dir: Optional[str] = None,
        lidar_point_count: int = 480,  # Add this parameter
    ):
        """
        Initialize the Oxford RobotCar Dataset.

        Args:
            root_dir: Root directory where the dataset is stored
            traversal: Traversal ID (date-time format)
            sensors: List of sensors to load (e.g., ["stereo_center", "lms_front"])
            use_radar: Whether to load radar data (if available)
            transform: Optional transform to apply to the data
            download: Whether to download the dataset if not found
            cache_dir: Directory to cache processed data
        """
        super().__init__()
        self.state_dim = 6
        self.obs_dim = 6
        self.ctrl_dim = 6

        self.root_dir = Path(root_dir)
        self.traversal = traversal
        self.use_radar = use_radar
        self.transform = transform
        self.lidar_point_count = lidar_point_count

        # Set default sensors if none provided
        if sensors is None:
            sensors = ["stereo_center", "lms_front", "gps_ins"]
        self.sensors = sensors

        # Setup cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path("/tmp/oxford_robotcar_cache")
        self.cache_dir.mkdir(exist_ok=True, parents=True)

        # Check if dataset exists and download if needed
        if not self._check_data_exists() and download:
            self._download_data()

        # Load and prepare dataset
        self.data = self._load_data()

        logging.getLogger(__name__).info(
            f"OxfordRobotCarDataset: traversal={traversal}, "
            f"sensors={sensors}, use_radar={use_radar}"
        )

        # caching
        sensors_str = "_".join(sorted(self.sensors))
        radar_str = "radar" if self.use_radar else "noradar"
        cache_name = f"robotcar_{self.traversal}_{sensors_str}_{radar_str}"
        cache_path = Path(".cache/data") / cache_name
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache(td.cachers.Pickle(cache_path))

    def _resample_pointcloud(self, points, target_size):
        """
        Resample point cloud to have a fixed number of points.

        Args:
            points: Point cloud tensor of shape [N, 3+] where N can vary
            target_size: Target number of points

        Returns:
            Resampled point cloud with exactly target_size points
        """
        if points.shape[0] == 0:
            # Return zeros if no points
            return torch.zeros((target_size, points.shape[1]), dtype=points.dtype)

        if points.shape[0] == target_size:
            # Already the right size
            return points

        if points.shape[0] > target_size:
            # Downsample: randomly select target_size points
            indices = torch.randperm(points.shape[0])[:target_size]
            return points[indices]
        else:
            # Upsample: randomly duplicate points
            # First, include all original points
            resampled = [points]

            # Then duplicate points until we reach the target size
            remaining = target_size - points.shape[0]
            if remaining > 0:
                # Randomly sample with replacement
                indices = torch.randint(0, points.shape[0], (remaining,))
                duplicates = points[indices]
                resampled.append(duplicates)

            return torch.cat(resampled, dim=0)

    def _check_data_exists(self) -> bool:
        """
        Check if dataset files exist at the expected location.

        Returns:
            bool: True if dataset files exist, False otherwise
        """
        traversal_dir = self.root_dir / self.traversal

        # Check essential files based on selected sensors
        for sensor in self.sensors:
            if "stereo" in sensor:
                camera, pos = sensor.split("_")
                timestamp_path = traversal_dir / camera / f"{camera}.timestamps"
                data_dir = traversal_dir / camera / pos
                if (
                    not timestamp_path.exists()
                    or not data_dir.exists()
                    or not any(data_dir.glob("*.png"))
                ):
                    return False

            elif "mono" in sensor:
                camera, pos = sensor.split("_")
                timestamp_path = traversal_dir / camera / f"{camera}.timestamps"
                data_dir = traversal_dir / camera
                if (
                    not timestamp_path.exists()
                    or not data_dir.exists()
                    or not any(data_dir.glob("*.png"))
                ):
                    return False

            elif any(
                lidar_type in sensor
                for lidar_type in ["lms_front", "lms_rear", "ldmrs"]
            ):
                lidar_type = sensor
                timestamp_path = traversal_dir / lidar_type / f"{lidar_type}.timestamps"
                data_dir = traversal_dir / lidar_type
                if (
                    not timestamp_path.exists()
                    or not data_dir.exists()
                    or not any(data_dir.glob("*.bin"))
                ):
                    return False

            elif "velodyne" in sensor:
                lidar_type = sensor
                timestamp_path = traversal_dir / lidar_type / f"{lidar_type}.timestamps"
                data_dir = traversal_dir / lidar_type
                if (
                    not timestamp_path.exists()
                    or not data_dir.exists()
                    or not (any(data_dir.glob("*.bin")) or any(data_dir.glob("*.png")))
                ):
                    return False

            elif "gps_ins" in sensor:
                timestamp_path = traversal_dir / "gps" / "ins.csv"
                if not timestamp_path.exists():
                    return False

        # Check radar data if requested
        if self.use_radar:
            radar_dir = traversal_dir / "radar"
            radar_timestamps = radar_dir / "radar.timestamps"
            if (
                not radar_dir.exists()
                or not radar_timestamps.exists()
                or not any(radar_dir.glob("*.png"))
            ):
                return False

        return True

    def _download_data(self):
        """
        Download dataset files or provide utility to help with manual download.

        The Oxford RobotCar Dataset requires registration and authentication,
        so this method provides a utility to help download the data.
        """
        # Create directories if they don't exist
        traversal_dir = self.root_dir / self.traversal
        traversal_dir.mkdir(exist_ok=True, parents=True)

        print("=" * 80)
        print("Oxford RobotCar Dataset Downloader")
        print("=" * 80)

        # Check if credentials file exists
        creds_file = Path.home() / ".oxford_robotcar_credentials"
        if creds_file.exists():
            with open(creds_file, "r") as f:
                creds = json.load(f)
                username = creds.get("username")
                password = creds.get("password")

            if username and password:
                print(f"Found credentials for user: {username}")
                print("Attempting to download dataset...")

                # Create session for authentication
                session = requests.Session()

                # Login to Oxford RobotCar website
                login_url = "https://robotcar-dataset.robots.ox.ac.uk/login/"
                login_data = {
                    "username": username,
                    "password": password,
                    "submit": "Login",
                }

                try:
                    login_response = session.post(login_url, data=login_data)
                    if "Login failed" in login_response.text:
                        print("Login failed. Please check your credentials.")
                        raise ValueError("Authentication failed")

                    # Download data for requested traversal
                    base_url = "https://robotcar-dataset.robots.ox.ac.uk/datasets/"

                    # Define file list - these vary by traversal but typically include:
                    file_types = [
                        "stereo.tar",
                        "mono.tar",
                        "gps.tar",
                        "lms.tar",
                        "ldmrs.tar",
                    ]

                    if self.use_radar:
                        file_types.append("radar.tar")

                    for file_type in file_types:
                        file_url = f"{base_url}{self.traversal}/{file_type}"
                        download_path = self.root_dir / f"{self.traversal}_{file_type}"

                        print(f"Downloading {file_url}")
                        response = session.get(file_url, stream=True)

                        if response.status_code == 200:
                            total_size = int(response.headers.get("content-length", 0))
                            with open(download_path, "wb") as f:
                                with tqdm(
                                    total=total_size, unit="B", unit_scale=True
                                ) as pbar:
                                    for chunk in response.iter_content(chunk_size=8192):
                                        if chunk:
                                            f.write(chunk)
                                            pbar.update(len(chunk))

                            # Extract file
                            print(f"Extracting {download_path}")
                            with tarfile.open(download_path) as tar:
                                tar.extractall(path=traversal_dir)

                            # Remove tar file after extraction
                            download_path.unlink()
                        else:
                            print(
                                f"Failed to download {file_url}: {response.status_code}"
                            )

                    print("Download completed. Please verify data integrity.")
                    return

                except Exception as e:
                    print(f"Error during download: {str(e)}")
                    print("Falling back to manual download instructions.")

        # If we get here, either credentials weren't found or download failed
        print(
            "\nThe Oxford RobotCar Dataset requires registration and manual download:"
        )
        print("1. Visit https://robotcar-dataset.robots.ox.ac.uk/")
        print("2. Register and request access to the dataset")
        print(
            "3. Download the requested traversal data and extract to your root directory"
        )
        print(f"   Expected path: {traversal_dir}")

        if self.use_radar:
            print("\nFor the Radar extension:")
            print("1. Visit http://ori.ox.ac.uk/datasets/radar-robotcar-dataset/")
            print("2. Register and download the radar data")
            print("3. Extract to the same traversal directory")
            print(f"   Expected path: {self.root_dir / self.traversal / 'radar'}")

        print(
            "\nOptionally, you can create a credentials file at ~/.oxford_robotcar_credentials:"
        )
        print('{"username": "your_username", "password": "your_password"}')
        print("This will enable automatic download in the future.")
        print("=" * 80)

        raise FileNotFoundError(
            f"Dataset not found at {traversal_dir}. "
            "Please follow the instructions above to download the dataset manually."
        )

    def _load_timestamps(self, path: Path) -> List[int]:
        """
        Load timestamps from a timestamps file.

        Args:
            path: Path to the timestamps file

        Returns:
            List of timestamps
        """
        if not path.exists():
            return []

        timestamps = []
        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 1:
                    timestamps.append(int(parts[0]))

        return timestamps

    def _load_data(self) -> Dict:
        """
        Load and preprocess the dataset.

        Returns:
            Dictionary containing processed data
        """
        # Check if cached data exists
        cache_file = (
            self.cache_dir
            / f"{self.traversal}_{'_'.join(self.sensors)}_radar{self.use_radar}.pt"
        )
        if cache_file.exists():
            print(f"Loading cached data from {cache_file}")
            return torch.load(cache_file)

        traversal_dir = self.root_dir / self.traversal
        data = {}

        # Load timestamps for each selected sensor
        timestamps_by_sensor = {}

        # Load camera data
        for sensor in self.sensors:
            if "stereo" in sensor:
                camera, pos = sensor.split("_")
                timestamp_path = traversal_dir / camera / f"{camera}.timestamps"
                timestamps_by_sensor[sensor] = self._load_timestamps(timestamp_path)
                data[f"{sensor}_paths"] = [
                    str(traversal_dir / camera / pos / f"{ts}.png")
                    for ts in timestamps_by_sensor[sensor]
                ]

            elif "mono" in sensor:
                camera, pos = sensor.split("_")
                timestamp_path = traversal_dir / camera / f"{camera}.timestamps"
                timestamps_by_sensor[sensor] = self._load_timestamps(timestamp_path)
                data[f"{sensor}_paths"] = [
                    str(traversal_dir / camera / f"{ts}.png")
                    for ts in timestamps_by_sensor[sensor]
                ]

            elif any(
                lidar_type in sensor
                for lidar_type in ["lms_front", "lms_rear", "ldmrs", "velodyne"]
            ):
                lidar_type = sensor
                timestamp_path = traversal_dir / lidar_type / f"{lidar_type}.timestamps"
                timestamps_by_sensor[sensor] = self._load_timestamps(timestamp_path)

                # Check for bin or png files
                if (
                    traversal_dir
                    / lidar_type
                    / f"{timestamps_by_sensor[sensor][0]}.bin"
                ).exists():
                    data[f"{sensor}_paths"] = [
                        str(traversal_dir / lidar_type / f"{ts}.bin")
                        for ts in timestamps_by_sensor[sensor]
                    ]
                else:
                    data[f"{sensor}_paths"] = [
                        str(traversal_dir / lidar_type / f"{ts}.png")
                        for ts in timestamps_by_sensor[sensor]
                    ]

            elif "gps_ins" in sensor:
                ins_path = traversal_dir / "gps" / "ins.csv"
                if ins_path.exists():
                    # Load INS data
                    ins_data = pd.read_csv(ins_path)
                    # # Simple preprocessing (timestamp, 3D position, 3D orientation)
                    # import pdb; pdb.set_trace()
                    # ins_data.columns = [
                    #     "timestamp",
                    #     "northing",
                    #     "easting",
                    #     "down",
                    #     "roll",
                    #     "pitch",
                    #     "yaw",
                    #     "status",
                    # ]
                    timestamps_by_sensor[sensor] = ins_data["timestamp"].tolist()
                    data[sensor] = ins_data

        # Load radar data if requested
        if self.use_radar:
            radar_timestamps_path = traversal_dir / "radar" / "radar.timestamps"
            if radar_timestamps_path.exists():
                timestamps_by_sensor["radar"] = self._load_timestamps(
                    radar_timestamps_path
                )
                data["radar_paths"] = [
                    str(traversal_dir / "radar" / f"{ts}.png")
                    for ts in timestamps_by_sensor["radar"]
                ]

        # Find common timestamps to synchronize data
        common_ts = self._find_common_timestamps(timestamps_by_sensor)
        data["common_timestamps"] = common_ts

        # Save cached data
        torch.save(data, cache_file)

        return data

    def _find_common_timestamps(
        self, timestamps_by_sensor: Dict[str, List[int]]
    ) -> List[Tuple[int, Dict[str, int]]]:
        """
        Find common timestamps across different sensors to synchronize data.

        Args:
            timestamps_by_sensor: Dictionary mapping sensor names to their timestamps

        Returns:
            List of (reference timestamp, {sensor: nearest timestamp}) pairs
        """
        if not timestamps_by_sensor:
            return []

        # Handle case with one sensor
        if len(timestamps_by_sensor) == 1:
            sensor = next(iter(timestamps_by_sensor))
            return [(ts, {sensor: ts}) for ts in timestamps_by_sensor[sensor]]

        # Find minimum timestamp distance for each sensor
        min_time_diffs = {
            sensor: np.median(np.diff(ts))
            for sensor, ts in timestamps_by_sensor.items()
            if len(ts) > 1
        }

        # Find the sensor with the lowest frequency (largest time difference)
        ref_sensor = max(min_time_diffs.items(), key=lambda x: x[1])[0]
        ref_timestamps = timestamps_by_sensor[ref_sensor]

        # Find the nearest timestamps in other sensors
        common_timestamps = []

        for ref_ts in ref_timestamps:
            match_found = True
            nearest_ts = {}

            for sensor, timestamps in timestamps_by_sensor.items():
                if sensor == ref_sensor:
                    nearest_ts[sensor] = ref_ts
                    continue

                # Find nearest timestamp
                if not timestamps:
                    match_found = False
                    break

                idx = np.searchsorted(timestamps, ref_ts)
                if idx == 0:
                    nearest = timestamps[0]
                elif idx == len(timestamps):
                    nearest = timestamps[-1]
                else:
                    # Check which is closer
                    if abs(timestamps[idx] - ref_ts) < abs(
                        timestamps[idx - 1] - ref_ts
                    ):
                        nearest = timestamps[idx]
                    else:
                        nearest = timestamps[idx - 1]

                # Check if timestamp is within acceptable range (half of min time diff)
                max_diff = min_time_diffs[sensor] * 0.5
                if abs(nearest - ref_ts) > max_diff:
                    match_found = False
                    break

                nearest_ts[sensor] = nearest

            if match_found:
                common_timestamps.append((ref_ts, nearest_ts))

        return common_timestamps

    def __len__(self):
        """Return the number of samples in the dataset"""
        if "common_timestamps" in self.data:
            return len(self.data["common_timestamps"])
        return 0

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)-1}]")

        ref_ts, nearest_ts = self.data["common_timestamps"][idx]
        gps_ins = self.data.get("gps_ins")
        if gps_ins is None or gps_ins.empty:
            raise RuntimeError("gps_ins not loaded. Use sensors=['gps_ins'].")

        # find closest row at this timestamp
        ts = nearest_ts.get("gps_ins", ref_ts)
        gi = int(np.argmin(np.abs(gps_ins["timestamp"].to_numpy() - ts)))
        row = gps_ins.iloc[gi]

        pos = torch.tensor(
            [row["northing"], row["easting"], row["down"]], dtype=torch.float32
        )
        eul = torch.tensor([row["roll"], row["pitch"], row["yaw"]], dtype=torch.float32)
        state = torch.cat([pos, eul], dim=-1)

        # controls = finite difference of state along the dataset index
        if idx == 0:
            ctrl = torch.zeros_like(state)
        else:
            ref_ts_prev, nearest_prev = self.data["common_timestamps"][idx - 1]
            ts_prev = nearest_prev.get("gps_ins", ref_ts_prev)
            gi_prev = int(np.argmin(np.abs(gps_ins["timestamp"].to_numpy() - ts_prev)))
            row_prev = gps_ins.iloc[gi_prev]
            pos_prev = torch.tensor(
                [row_prev["northing"], row_prev["easting"], row_prev["down"]],
                dtype=torch.float32,
            )
            eul_prev = torch.tensor(
                [row_prev["roll"], row_prev["pitch"], row_prev["yaw"]],
                dtype=torch.float32,
            )
            ctrl = torch.cat([pos - pos_prev, eul - eul_prev], dim=-1)

        # observations: same as state for a clean minimal feed
        obs = state.clone()

        # return with a trivial time axis, so windowing later is straightforward
        return {
            "states": state.view(1, -1),
            "observations": obs.view(1, -1),
            "controls": ctrl.view(1, -1),
        }

    def _get_single_item(self, idx):
        """
        Get a single frame from the dataset.

        Args:
            idx: Index of the frame

        Returns:
            Dictionary containing the frame data
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)-1}]")

        # Get timestamp information
        ref_ts, nearest_ts = self.data["common_timestamps"][idx]

        # Initialize sample dictionary
        sample = {"timestamp": ref_ts}

        # Load images
        for sensor in self.sensors:
            if "stereo" in sensor or "mono" in sensor:
                sensor_ts = nearest_ts.get(sensor, ref_ts)
                img_path = None

                # Find the image path
                for path in self.data.get(f"{sensor}_paths", []):
                    if str(sensor_ts) in path:
                        img_path = path
                        break

                if img_path and os.path.exists(img_path):
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        if self.transform:
                            img = self.transform(img)
                        img_tensor = torch.from_numpy(img).float() / 255.0
                        sample[f"{sensor}_image"] = img_tensor

        # Load LIDAR data
        for sensor in self.sensors:
            if any(
                lidar_type in sensor
                for lidar_type in ["lms_front", "lms_rear", "ldmrs", "velodyne"]
            ):
                sensor_ts = nearest_ts.get(sensor, ref_ts)
                lidar_path = None

                # Find the lidar path
                for path in self.data.get(f"{sensor}_paths", []):
                    if str(sensor_ts) in path:
                        lidar_path = path
                        break

                if lidar_path and os.path.exists(lidar_path):
                    # Load based on file extension
                    if lidar_path.endswith(".bin"):
                        with open(lidar_path, "rb") as f:
                            lidar_data = np.fromfile(f, dtype=np.double)
                            lidar_data = lidar_data.reshape((-1, 3))
                    elif lidar_path.endswith(".png"):
                        # For velodyne data stored as PNG
                        lidar_data = cv2.imread(lidar_path, cv2.IMREAD_UNCHANGED)

                    lidar_tensor = torch.from_numpy(lidar_data).float()

                    # Resample to fixed size - ADD THIS LINE
                    lidar_tensor = self._resample_pointcloud(
                        lidar_tensor, self.lidar_point_count
                    )

                    sample[f"{sensor}_points"] = lidar_tensor

        # Load GPS/INS data
        if "gps_ins" in self.sensors:
            gps_ins_data = self.data.get("gps_ins")
            if gps_ins_data is not None:
                ts = nearest_ts.get("gps_ins", ref_ts)
                # Find closest timestamp
                closest_idx = np.argmin(np.abs(gps_ins_data["timestamp"] - ts))
                row = gps_ins_data.iloc[closest_idx]

                # Position and orientation
                pos = torch.tensor(
                    [row["northing"], row["easting"], row["down"]], dtype=torch.float32
                )
                ori = torch.tensor(
                    [row["roll"], row["pitch"], row["yaw"]], dtype=torch.float32
                )

                sample["position"] = pos
                sample["orientation"] = ori

        # Load radar data if requested
        if self.use_radar:
            radar_ts = nearest_ts.get("radar", ref_ts)
            radar_path = None

            # Find the radar path
            for path in self.data.get("radar_paths", []):
                if str(radar_ts) in path:
                    radar_path = path
                    break

            if radar_path and os.path.exists(radar_path):
                radar_data = cv2.imread(radar_path, cv2.IMREAD_UNCHANGED)
                radar_tensor = torch.from_numpy(radar_data).float()
                sample["radar_scan"] = radar_tensor

        return sample

    def _get_data(self, indices):
        """
        Get multiple items by their indices.

        Args:
            indices: List of indices to retrieve

        Returns:
            Dictionary with batched data from the specified indices
        """
        # Get individual samples
        samples = [self._get_single_item(idx) for idx in indices]

        # Combine samples into a single batch
        result = {}

        # Process all keys in the first sample
        for key in samples[0].keys():
            if key == "timestamp":
                # Store timestamps as list
                result[key] = [s[key] for s in samples]
                continue

            # Try to stack tensor data
            if isinstance(samples[0][key], torch.Tensor):
                try:
                    result[key] = torch.stack([s[key] for s in samples])
                except:
                    # For data with inconsistent shapes (e.g., point clouds)
                    result[key] = [s[key] for s in samples]
            else:
                # For non-tensor data
                result[key] = [s[key] for s in samples]

        return result
