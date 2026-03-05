import os
import pickle
import tarfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
import torch
import torchdatasets as td
from tqdm import tqdm

TMP_PREFIX = f"/tmp/{os.environ.get('USER', 'cache')}"


class NCLTDataset(td.Dataset):
    BASE_URL = "https://s3.us-east-2.amazonaws.com/nclt.perl.engin.umich.edu"

    # Define dates for the dataset
    DATES = [
        "2012-01-08",
        "2012-01-15",
        "2012-01-22",
        "2012-02-02",
        "2012-02-04",
        "2012-02-05",
        "2012-02-12",
        "2012-02-18",
        "2012-02-19",
        "2012-03-17",
        "2012-03-25",
        "2012-03-31",
        "2012-04-29",
        "2012-05-11",
        "2012-05-26",
        "2012-06-15",
        "2012-08-04",
        "2012-08-20",
        "2012-09-28",
        "2012-10-28",
        "2012-11-04",
        "2012-11-16",
        "2012-11-17",
        "2012-12-01",
        "2013-01-10",
        "2013-02-23",
        "2013-04-05",
    ]

    def __init__(
        self,
        root_dir: str = "./data/NCLT",
        download: bool = True,
        date: Optional[str] = "2012-01-22",
        transform=None,
    ):
        super().__init__()

        self.state_dim = 3
        self.obs_dim = 6
        self.ctrl_dim = 6

        self.root_dir = Path(root_dir)
        self.transform = transform

        # Create directories
        self.data_dir = self.root_dir / "nclt_data"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        if download and not self._check_data_exists(date):
            self._download_data(date)

        # Load data (with caching)
        self.data = self._load_data(date)

        cache_path = Path(".cache/data")
        cache_path = cache_path / f"nclt_{date}"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache(td.cachers.Pickle(cache_path))

    def _check_data_exists(self, date=None, velodyne=False):
        """
        Check if NCLT dataset files exist at the expected location

        Args:
            date (str, optional): Specific date to check. If None, checks all dates.

        Returns:
            bool: True if all files for the specified date(s) exist, False otherwise
        """
        # If a specific date is given, only check that date
        dates_to_check = [date] if date else self.DATES

        for date in dates_to_check:
            # 1. Check for sensor data files in date directory
            date_dir = self.data_dir / date
            odom_file = date_dir / "odometry_mu.csv"
            odom_cov_file = date_dir / "odometry_cov.csv"
            ms25_file = date_dir / "ms25.csv"
            ms25_euler_file = date_dir / "ms25_euler.csv"
            gps_file = date_dir / "gps.csv"

            # 2. Check for ground truth file
            gt_file = self.data_dir / f"groundtruth_{date}.csv"

            # 3. Check all these files exist
            required_files = [
                odom_file,
                odom_cov_file,
                ms25_file,
                ms25_euler_file,
                gps_file,
                gt_file,
            ]

            if not all(f.exists() for f in required_files):
                print(f"Missing files for date {date}")
                return False

            # 4. Check for velodyne data
            vel_sync_dir = date_dir / "velodyne_data" / "velodyne_sync"
            if velodyne and (
                not vel_sync_dir.exists() or not any(vel_sync_dir.glob("*.bin"))
            ):
                print(f"Missing velodyne data for date {date}")
                return False

        return True

    def _download_data(self, date=None):
        """
        Download NCLT dataset files

        Args:
            date (str, optional): Specific date to download. If None, downloads all dates.
        """
        # If a specific date is given, only download that date
        dates_to_download = [date] if date else self.DATES

        print("Checking required data files...")
        for date in tqdm(dates_to_download, desc="Checking datasets"):
            # First check if all extracted data exists
            if self._check_data_exists(date):
                print(f"Data for {date} already exists, skipping.")
                continue

            # Define prefixes and URLs
            sensor_tar_file = f"{date}_sen.tar.gz"
            sensor_url = f"{self.BASE_URL}/sensor_data/{sensor_tar_file}"
            sensor_tar_path = self.data_dir / sensor_tar_file

            gt_csv_file = f"groundtruth_{date}.csv"
            gt_csv_url = f"{self.BASE_URL}/ground_truth/{gt_csv_file}"
            gt_csv_path = self.data_dir / gt_csv_file

            vel_tar_file = f"{date}_vel.tar.gz"
            vel_url = f"{self.BASE_URL}/velodyne_data/{vel_tar_file}"
            vel_tar_path = self.data_dir / vel_tar_file

            # Ensure date directory exists
            date_dir = self.data_dir / date
            date_dir.mkdir(parents=True, exist_ok=True)

            # Download odometry data
            odom_data_file = date_dir / "odometry_mu.csv"
            if not odom_data_file.exists():
                # Download tar file if needed
                if not sensor_tar_path.exists():
                    print(f"\nDownloading {sensor_tar_file}...")
                    self._download_file(sensor_url, sensor_tar_path)

                # Extract tar file regardless (maybe it was interrupted last time)
                print(f"\nExtracting {sensor_tar_file}...")
                self._extract_tar(sensor_tar_path, output_dir=self.data_dir)

            # Download ground truth if needed
            if not gt_csv_path.exists():
                print(f"\nDownloading {gt_csv_file}...")
                self._download_file(gt_csv_url, gt_csv_path)

            # Download and extract velodyne data if needed
            vel_date_dir = date_dir / "velodyne_data"
            vel_sync_dir = vel_date_dir / "velodyne_sync"
            if not vel_sync_dir.exists() or not any(vel_sync_dir.glob("*.bin")):
                # Download tar file if needed
                if not vel_tar_path.exists():
                    print(f"\nDownloading {vel_tar_file}...")
                    self._download_file(vel_url, vel_tar_path)

                # Extract tar file regardless (maybe it was interrupted last time)
                print(f"\nExtracting {vel_tar_file}...")
                self._extract_tar(vel_tar_path, output_dir=self.data_dir)

            # Verify download was successful
            if not self._check_data_exists(date):
                print(
                    f"Warning: Some files for {date} are still missing after download."
                )

    def _extract_tar(self, tar_path, output_dir=None):
        """
        Extract tar file with progress bar to a specific directory

        Args:
            tar_path (Path): Path to the tar file
            output_dir (Path, optional): Directory to extract to. Defaults to date directory.
        """
        output_dir = output_dir if output_dir is not None else self.data_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        with tarfile.open(tar_path, "r:gz") as tar:
            members = tar.getmembers()
            for member in tqdm(
                members, desc=f"Extracting {tar_path.name} into {output_dir}"
            ):
                tar.extract(member, path=output_dir)
                print(f"Extracted {member.name} to {output_dir}")

    def _download_file(self, url: str, output_path: Path):
        """Download file from URL"""
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        else:
            raise Exception(f"Failed to download {url}")

    def _load_data(self, date=None):
        """
        Load data with caching

        Args:
            date (str, optional): Specific date to load. If None, loads all dates.

        Returns:
            DataFrame: All data points
        """
        # If a specific date is given, only load that date
        dates_to_load = [date] if date else self.DATES

        # Create a unique cache file name based on dates
        cache_prefix = Path(f"{TMP_PREFIX}/data/nclt")
        cache_prefix.mkdir(parents=True, exist_ok=True)

        all_data = []

        # Define column data types
        dtype_dict = {
            0: np.int64,  # utime
            1: np.float64,  # x
            2: np.float64,  # y
            3: np.float64,  # z
            4: np.float64,  # phi
            5: np.float64,  # theta
            6: np.float64,  # psi
        }

        for date in tqdm(dates_to_load, desc="Processing dates"):
            cache_path = cache_prefix / f"full_data_{date}.pkl"
            if cache_path.exists():
                print(f"Loading data for {date} from cache: {cache_path}")
                with open(cache_path, "rb") as f:
                    date_data = pickle.load(f)
                    all_data.append(date_data)
            else:
                try:
                    # Load odometry data (control inputs)
                    odom_path = self.data_dir / date / "odometry_mu.csv"
                    odometry_data = pd.read_csv(
                        odom_path, header=None, low_memory=False
                    )
                    odometry_data = (
                        odometry_data.astype(float).dropna().astype(dtype_dict)
                    )
                    odometry_data.columns = [
                        "utime",
                        "x",
                        "y",
                        "z",
                        "phi",
                        "theta",
                        "psi",
                    ]

                    # Load ground truth
                    gt_path = self.data_dir / f"groundtruth_{date}.csv"
                    gt_data = pd.read_csv(gt_path, header=None, low_memory=False)
                    gt_data = gt_data.astype(float).dropna().astype(dtype_dict)
                    gt_data.columns = ["utime", "x", "y", "z", "phi", "theta", "psi"]

                    # Merge based on timestamp
                    merged_data = pd.merge_asof(
                        odometry_data.sort_values("utime"),
                        gt_data.sort_values("utime"),
                        on="utime",
                        suffixes=("_odom", "_gt"),
                    )

                    # Calculate control inputs (derivative)
                    merged_data["dx_odom"] = 0.0
                    merged_data["dy_odom"] = 0.0
                    merged_data["dz_odom"] = 0.0
                    merged_data["dphi_odom"] = 0.0
                    merged_data["dtheta_odom"] = 0.0
                    merged_data["dpsi_odom"] = 0.0

                    # Calculate the deltas for all rows after the first
                    for i in range(1, len(merged_data)):
                        merged_data.loc[merged_data.index[i], "dx_odom"] = (
                            merged_data.iloc[i]["x_odom"]
                            - merged_data.iloc[i - 1]["x_odom"]
                        )
                        merged_data.loc[merged_data.index[i], "dy_odom"] = (
                            merged_data.iloc[i]["y_odom"]
                            - merged_data.iloc[i - 1]["y_odom"]
                        )
                        merged_data.loc[merged_data.index[i], "dz_odom"] = (
                            merged_data.iloc[i]["z_odom"]
                            - merged_data.iloc[i - 1]["z_odom"]
                        )
                        merged_data.loc[merged_data.index[i], "dphi_odom"] = (
                            merged_data.iloc[i]["phi_odom"]
                            - merged_data.iloc[i - 1]["phi_odom"]
                        )
                        merged_data.loc[merged_data.index[i], "dtheta_odom"] = (
                            merged_data.iloc[i]["theta_odom"]
                            - merged_data.iloc[i - 1]["theta_odom"]
                        )
                        merged_data.loc[merged_data.index[i], "dpsi_odom"] = (
                            merged_data.iloc[i]["psi_odom"]
                            - merged_data.iloc[i - 1]["psi_odom"]
                        )

                    # Add date information
                    merged_data["date"] = date

                    # Save to cache
                    with open(cache_path, "wb") as f:
                        pickle.dump(merged_data, f)

                    all_data.append(merged_data)
                    print(f"Processed {len(merged_data)} data points for {date}")
                except Exception as e:
                    print(f"Error loading {date}: {e}")
                    continue

        # Combine all data from different dates
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        else:
            return pd.DataFrame()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            # Handle single index
            row = self.data.iloc[idx]
            return self._process_single_item(row)
        else:
            # Handle slice objects (for sequence samplers)
            rows = self.data.iloc[idx]
            return self._process_data(rows)

    # change _process_single_item to return the minimal dict:
    def _process_single_item(self, row):
        keys3d = ["x", "y", "z", "phi", "theta", "psi"]
        keys2d = ["x", "y", "theta"]

        odom = torch.tensor([row[f"{k}_odom"] for k in keys3d], dtype=torch.float32)
        control = torch.tensor([row[f"d{k}_odom"] for k in keys3d], dtype=torch.float32)
        gt_poses = torch.tensor([row[f"{k}_gt"] for k in keys2d], dtype=torch.float32)

        return {
            "states": gt_poses.view(1, -1),
            "observations": odom.view(1, -1),
            "controls": control.view(1, -1),
        }

    # and _process_data for slices:
    def _process_data(self, rows):
        keys3d = ["x", "y", "z", "phi", "theta", "psi"]
        keys2d = ["x", "y", "theta"]

        odom = torch.tensor(
            [[r[f"{k}_odom"] for k in keys3d] for _, r in rows.iterrows()],
            dtype=torch.float32,
        )
        control = torch.tensor(
            [[r[f"d{k}_odom"] for k in keys3d] for _, r in rows.iterrows()],
            dtype=torch.float32,
        )
        gt_poses = torch.tensor(
            [[r[f"{k}_gt"] for k in keys2d] for _, r in rows.iterrows()],
            dtype=torch.float32,
        )

        return {"states": gt_poses, "observations": odom, "controls": control}
