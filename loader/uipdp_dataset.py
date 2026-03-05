# loader.py
import hashlib
import logging
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torchdatasets as td

from .articulate import ParametricModel
from .articulate.math import (
    RotationRepresentation,
    rotation_matrix_to_euler_angle,
    to_rotation_matrix,
)
from .utils import rotation_matrix_to_quaternion

ROT_DIM = {
    "euler": 3,
    "quat": 4,
    "matrix": 9,
}


# from plot_imu_predictions import plot_imu_predictions
# from plot_uwb_predictions import plot_uwb_predictions
def get_global_acc(v, frame_rate, smooth_n=4):
    r"""
    Synthesize accelerations from vertex positions.
    """
    mid = smooth_n // 2
    acc = torch.stack(
        [
            (v[i] + v[i + 2] - 2 * v[i + 1]) * frame_rate**2
            for i in range(0, v.shape[0] - 2)
        ]
    )
    acc = torch.cat((torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1])))
    if mid != 0:
        acc[smooth_n:-smooth_n] = torch.stack(
            [
                (v[i] + v[i + smooth_n * 2] - 2 * v[i + smooth_n])
                * frame_rate**2
                / smooth_n**2
                for i in range(0, v.shape[0] - smooth_n * 2)
            ]
        )
    return acc


def get_non_root_joint_pos(glb_joint_pos, glb_root_ori):
    _r = glb_joint_pos[:, 1:, :] - glb_joint_pos[:, joint_set.root, :]
    return torch.einsum("bij,bjk->bik", _r, glb_root_ori.view(-1, 3, 3))


def get_leaf_joint(glb_joint_pos, glb_root_ori):
    # get leaf joint position relative to root
    _r = glb_joint_pos[:, joint_set.leaf, :] - glb_joint_pos[:, joint_set.root]
    return torch.einsum("bij,bjk->bik", _r, glb_root_ori.view(-1, 3, 3))


def get_6d_rotation(global_rotmat, glb_root_ori):
    # get 6d rotation relative to root
    N_f = global_rotmat.size(0)
    global_rotmat = torch.einsum(
        "abij,abjk->abik",
        glb_root_ori.view(-1, 1, 3, 3).transpose(2, 3),
        global_rotmat,
    )
    return (
        global_rotmat.view(N_f, -1, 3, 3)[:, :, :, :2]
        .transpose(2, 3)
        .contiguous()
        .view(N_f, -1, 6)
    )


def get_joint_vel(glb_joint_pos, glb_root_ori):
    N_f = glb_joint_pos.size(0)
    joint_vel = torch.zeros_like(glb_joint_pos)
    joint_vel[1:] = (glb_joint_pos[1:, :, :] - glb_joint_pos[:-1, :, :]) * 20
    joint_vel = torch.einsum("bij,bjk->bik", joint_vel, glb_root_ori.view(N_f, 3, 3))
    joint_vel[0] = joint_vel[1]
    return joint_vel


def get_contact_points(glb_joint_pos, th=0.0125):
    feet_contact = torch.zeros(glb_joint_pos.size(0), 2)
    feet_contact[1:, :] = (
        torch.linalg.norm(
            glb_joint_pos[1:, 10:12, :] - glb_joint_pos[:-1, 10:12, :], dim=2
        )
        < th
    )
    feet_contact[0] = feet_contact[1]
    feet_contact = feet_contact.to(torch.int)
    return feet_contact


def get_acc_sum(acc, window_size=40, down_scale=15):
    b = torch.cumsum(acc.view(-1, 18), dim=0)
    b[window_size:, :] = b[window_size:, :] - b[:-window_size, :]
    b = b / down_scale  # down scale to acc scale
    return b.view_as(acc)


def get_local_imu(glb_acc, glb_rot):
    glb_acc = glb_acc.view(-1, 6, 3)
    glb_rot = glb_rot.view(-1, 6, 3, 3)
    acc = torch.cat((glb_acc[:, :5] - glb_acc[:, 5:], glb_acc[:, 5:]), dim=1).bmm(
        glb_rot[:, -1]
    )
    ori = torch.cat(
        (glb_rot[:, 5:].transpose(2, 3).matmul(glb_rot[:, :5]), glb_rot[:, 5:]),
        dim=1,
    )

    return acc, ori


def preprocess_pose_trans(pose, shape=None, tran=None, device="cpu"):
    pose = to_rotation_matrix(pose.to(device), RotationRepresentation.AXIS_ANGLE).view(
        pose.shape[0], -1
    )
    shape = shape.to(device) if shape is not None else shape
    tran = tran.to(device) if tran is not None else tran
    return pose, shape, tran


def hash_tensor_dict(d: "dict[str, list[torch.Tensor]]") -> str:
    m = hashlib.sha256()
    for key in sorted(d):  # ensure deterministic order
        m.update(key.encode())
        for tensor in d[key]:
            if isinstance(tensor, torch.Tensor):
                m.update(tensor.cpu().numpy().tobytes())  # convert to bytes
                m.update(str(tensor.dtype).encode())
                m.update(str(tensor.shape).encode())
            else:
                m.update(tensor.encode())
    return m.hexdigest()


class joint_set:
    root = [0]
    leaf = [4, 5, 12, 20, 21]
    full = list(range(1, 24))
    reduced = [1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]
    ignored = [0, 7, 8, 10, 11, 20, 21, 22, 23]
    n_leaf = len(leaf)
    n_full = len(full)
    n_reduced = len(reduced)
    n_ignored = len(ignored)


###############################################
# Helper Function for UWB Processing
###############################################


def convert_batch_to_euler(rotation_matrices):
    """
    Convert a batch of rotation matrices to Euler angles using articulate.math.

    Args:
        rotation_matrices: Tensor of shape (T, num_devices, 3, 3)

    Returns:
        Tensor of shape (T, num_devices, 3) with Euler angles in radians
    """
    # Get shape information
    T, num_devices = rotation_matrices.shape[:2]

    # Reshape to (T*num_devices, 3, 3)
    rot_reshaped = rotation_matrices.reshape(-1, 3, 3)

    # Convert to Euler angles using articulate.math
    euler_angles = rotation_matrix_to_euler_angle(rot_reshaped, seq="XYZ")

    # Reshape back to (T, num_devices, 3)
    euler_angles = euler_angles.reshape(T, num_devices, 3)

    return euler_angles


def extract_upper_triangle(matrix):
    """
    Extracts the strictly upper-triangular elements (excluding the diagonal)
    from a matrix.

    Args:
      - matrix: Tensor of shape (..., 6, 6)

    Returns:
      A tensor of shape (..., 15) containing the strictly upper-triangular elements.
    """
    indices = torch.triu_indices(6, 6, offset=1, device=matrix.device)
    return matrix[..., indices[0], indices[1]]


def generate_mock_data(num_samples: int, seq_len: int, x_dim_total: int):
    # Generate smooth target signals.
    smooth_data = torch.zeros(num_samples, seq_len, x_dim_total)
    for i in range(num_samples):
        # Generate a small-noise sequence and compute cumulative sum.
        noise = torch.randn(seq_len, x_dim_total) * 0.1
        smooth_sample = torch.cumsum(noise, dim=0)
        # Apply a simple smoothing kernel.
        kernel = np.ones(3) / 3.0
        smooth_sample_np = smooth_sample.numpy()
        for ch in range(x_dim_total):
            smooth_sample_np[:, ch] = np.convolve(
                smooth_sample_np[:, ch], kernel, mode="same"
            )
        smooth_sample = torch.tensor(smooth_sample_np, dtype=torch.float)
        smooth_data[i] = smooth_sample

    # Create raw sensor data by adding extra noise.
    raw_data = smooth_data + torch.randn_like(smooth_data) * 0.5
    noise_data = torch.randn_like(smooth_data) * torch.exp(
        torch.randn_like(smooth_data)
    )
    noise_gate = torch.rand_like(smooth_data)
    raw_data = (1 - noise_gate) * raw_data + noise_gate * noise_data

    return raw_data, smooth_data


###############################################
# Combined Dataset Class
###############################################


class UWBIMUDataset(td.Dataset):
    """
    Combined dataset that returns both IMU and UWB data.

    For each sample, returns a tuple:
      (imu_input, imu_target, uwb_input, uwb_target, length)

    - imu_input: Tensor of shape (seq_len, 72)
    - imu_target: Tensor of shape (seq_len, 72)
    - uwb_input: Tensor of shape (seq_len, 15) obtained by extracting the strictly upper-triangular
                 elements of the original 6x6 UWB matrix.
    - uwb_target: Tensor of shape (seq_len, 15) processed similarly.
    - length: integer, the actual sequence length extracted.
    """

    TEST_FNAMES = [
        # "00_session1_freestyle",
        # "00_session2_0",
        "00_session2_1",
        "00_session1_1",
        "00_session1_0",
        # "02_session1_0",
        "02_session2_1",
        "02_session2_0",
        "02_session1_1",
        "03_session1_0",
        "03_session1_1",
        "03_session1_freestyle",
        "03_session2_0",
        "03_session2_1",
        "03_session1_2",
    ]

    TRAIN_FNAMES = [
        "04_session1_0",
        "04_session1_freestyle",
        "04_session2_1",
        "04_session1_1",
        "04_session2_0",
        "05_session2_1",
        "05_session1_2",
        "05_session2_0",
        "05_session1_freestyle",
        "05_session1_1",
        "05_session1_0",
        # "06_session2_1",
        "06_session2_0",
        "06_session1_freestyle",
        "07_session2_0",
        "07_session1_1",
        "07_session2_1",
        # "09_session1_1",
        "09_session1_0",
        "09_session2_0",
        "09_session2_1",
        "09_session1_freestyle",
    ]

    def __init__(
        self,
        # root_dir: str,
        # session_name: str,
        root_dir: str = "./data/UIP_DP",
        data_file: str = "",
        session_name: str = "00_session2_1",
        input_type: str = "raw",
        label_type: str = "ekf",
        uwb_repr: str = "full",
        rot_repr: str = "quat",
        train: bool = False,
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        assert uwb_repr in ["upper_triangle", "full"]
        assert rot_repr in ["quat", "euler", "matrix"]

        if data_file:
            assert os.path.isfile(self.root_dir / data_file), (
                f"File Does not exists: {self.root_dir / data_file}."
            )
        elif session_name in self.TRAIN_FNAMES:
            data_file = "raw_train.pt"
        elif session_name in self.TEST_FNAMES:
            data_file = "raw_test.pt"
        else:
            raise ValueError(f"No such session name: {session_name}")

        # Load data from the provided file path.
        if not self._check_data_exists(session_name):
            self._download_data(data_file)

        self.train = train
        self.uwb_repr = uwb_repr
        self.rot_repr = rot_repr
        self.rot_dim = ROT_DIM[rot_repr]
        self.num_devices = 6

        # loading data
        session_data = self._load_data_session(
            session_name, input_type=input_type, label_type=label_type
        )
        self.imu_inputs = session_data["imu_data"]
        self.uwb_inputs = session_data["uwb_data"]
        self.imu_targets = session_data["imu_gt"]
        self.uwb_targets = session_data["uwb_gt"]
        self.acc_global = session_data["imu_ctrl"]
        self.fname = session_data["fname"]
        import pdb

        pdb.set_trace()

        logger = logging.getLogger(__name__)
        logger.info(
            f"{self.__class__.__name__} session_name={session_name} input_type={input_type}, label_type={label_type}, uwb_repr={self.uwb_repr} train={self.train}."
        )
        train_str = "train" if self.train else "test"
        cache_name = f"uwbimu_{session_name}_{input_type}_{label_type}_{self.uwb_repr}_{train_str}"
        cache_path = Path(".cache/data") / cache_name
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache(td.cachers.Pickle(cache_path))

    def __len__(self):
        return len(self.imu_inputs)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        uwb_inp = self.uwb_inputs[index].view(
            -1, self.num_devices, self.num_devices
        )  # (seq_len, 6, 6)
        uwb_tgt = self.uwb_targets[index].view(
            -1, self.num_devices, self.num_devices
        )  # (seq_len, 6, 6)

        if self.uwb_repr == "upper_triangle":
            # For each time step, extract the strictly upper-triangular part (vector of length 15).
            uwb_input = torch.stack(
                [extract_upper_triangle(frame) for frame in uwb_inp], dim=0
            ).view(
                -1, (self.num_devices * (self.num_devices - 1)) // 2
            )  # (seq_len, 15)

            uwb_target = torch.stack(
                [extract_upper_triangle(frame) for frame in uwb_tgt], dim=0
            ).view(
                -1, (self.num_devices * (self.num_devices - 1)) // 2
            )  # (seq_len, 15)
        else:
            uwb_input = uwb_inp.view(
                -1, self.num_devices * self.num_devices
            )  # (seq_len, 36)
            uwb_target = uwb_tgt.view(
                -1, self.num_devices * self.num_devices
            )  # (seq_len, 36)

        imu_features = 3 + self.rot_dim
        imu_input = self.imu_inputs[index].view(
            -1, self.num_devices * imu_features
        )  # (seq_len, 72)
        imu_target = self.imu_targets[index].view(
            -1, self.num_devices * imu_features
        )  # (seq_len, 72)
        imu_control = self.acc_global[index].view(-1, self.num_devices * imu_features)

        return {
            "states": torch.cat((imu_target, uwb_target), dim=1),
            "controls": imu_control,
            "observations": torch.cat((imu_input, uwb_input), dim=1),
        }

        return {
            "imu_input": imu_input,
            "imu_target": imu_target,
            "imu_control": imu_control,
            "uwb_input": uwb_input,
            "uwb_target": uwb_target,
            "filename": self.fname,
        }

    def _check_data_exists(self, session_name: str = "all") -> bool:
        sessions_dir = self.root_dir / "sessions"
        if not os.path.isdir(sessions_dir):
            return False

        if session_name == "all":
            sessions = set(self.TRAIN_FNAMES).union(self.TEST_FNAMES)
        else:
            sessions = {session_name}

        all_session_paths = map(sessions_dir.__truediv__, sessions)
        return all(map(os.path.isfile, all_session_paths))

    def _load_data_session(
        self, session_name: str, input_type: str = "raw", label_type: str = "ekf"
    ) -> Dict[str, torch.Tensor]:
        session_file = self.root_dir / "sessions" / f"{session_name}.pt"
        seq_data = torch.load(session_file, weights_only=True)
        # print(
        #     f"{self.__class__.__name__}: session {session_name} "
        #     f"is loaded from {session_file}."
        # )
        if input_type == "raw":
            acc_data = seq_data["f_acc_raw"]
            ori_data = seq_data["gyro_raw"]
        elif input_type == "ekf":
            acc_data = seq_data["imu_acc"]
            ori_data = seq_data["imu_ori"]
        else:
            raise ValueError(f"Invalid input_type: {input_type}.")

        if label_type == "ekf":
            acc_gt = seq_data["imu_acc"]
            ori_gt = seq_data["imu_ori"]
        else:
            raise ValueError(f"Invalid label_type: {label_type}.")

        acc_global = seq_data["glb_acc"]
        ori_global = seq_data["glb_ori"]
        uwb_data = seq_data["uwb_m"]
        uwb_gt = seq_data["uwb_gt"]

        if self.rot_repr == "quat" and ori_global.shape[-2:] == (3, 3):
            SeqLen, NumDevices = ori_global.shape[:-2]
            ori_global = rotation_matrix_to_quaternion(
                ori_global.view(SeqLen * NumDevices, 3, 3)
            )
            ori_global = ori_global.view(SeqLen, NumDevices, 4)

        if self.rot_repr == "quat" and ori_data.shape[-2:] == (3, 3):
            SeqLen, NumDevices = ori_data.shape[:-2]
            ori_data = rotation_matrix_to_quaternion(
                ori_data.view(SeqLen * NumDevices, 3, 3)
            )
            ori_data = ori_data.view(SeqLen, NumDevices, 4)

        if self.rot_repr == "quat" and ori_gt.shape[-2:] == (3, 3):
            SeqLen, NumDevices = ori_gt.shape[:-2]
            ori_gt = rotation_matrix_to_quaternion(
                ori_gt.view(SeqLen * NumDevices, 3, 3)
            )
            ori_gt = ori_gt.view(SeqLen, NumDevices, 4)

        return dict(
            imu_data=torch.cat((acc_data.flatten(1), ori_data.flatten(1)), dim=-1),
            imu_ctrl=torch.cat((acc_global.flatten(1), ori_global.flatten(1)), dim=-1),
            imu_gt=torch.cat((acc_gt.flatten(1), ori_gt.flatten(1)), dim=-1),
            uwb_data=uwb_data.flatten(1),
            uwb_gt=uwb_gt.flatten(1),
            fname=seq_data["fnames"],
        )

    def _download_data(self, data_file: str = "train.pt"):
        raw_data = torch.load(self.root_dir / data_file, weights_only=True)
        processed_data = defaultdict(list)
        device = torch.device("cpu")
        normalize_uwb = True
        flatten_uwb = False
        remove_node = -1

        model = ParametricModel(
            f"{self.root_dir}/basicmodel_m_lbs_10_207_0_v1.0.0.pkl",
            use_pose_blendshape=False,
            device=device,
        )

        IMU_NUM = 6
        ji_mask = torch.tensor([18, 19, 4, 5, 15, 0])
        mesh_mask = torch.tensor([1961, 5424, 1067, 4553, 457, 3021])

        MESH_DATA_PREFIX = "/tmp/adhd/mesh-data-" + hash_tensor_dict(raw_data)
        os.makedirs(MESH_DATA_PREFIX, exist_ok=True)

        os.makedirs(self.root_dir / "sessions", exist_ok=True)
        pattern = re.compile(r"(\d{2}_session\d+_(?:\d+|freestyle))")

        for seq_id in range(len(raw_data["fnames"])):
            seq_kwargs = dict()

            session_name = pattern.search(raw_data["fnames"][seq_id]).group(0)
            session_file = self.root_dir / "sessions" / f"{session_name}.pt"

            if not session_file.exists():
                seq_kwargs["glb_ori"] = raw_data["ori"][seq_id]
                seq_kwargs["glb_acc"] = raw_data["acc"][seq_id]

                if "f_acc_raw" in raw_data:
                    seq_kwargs["f_acc_raw"] = raw_data["f_acc_raw"][seq_id]
                if "gyro_raw" in raw_data:
                    seq_kwargs["gyro_raw"] = raw_data["gyro_raw"][seq_id]

                seq_kwargs["imu_acc"], seq_kwargs["imu_ori"] = get_local_imu(
                    raw_data["acc"][seq_id], raw_data["ori"][seq_id]
                )
                # Forward kinematics
                pose_p, shape_p, tran_p = preprocess_pose_trans(
                    raw_data["pose"][seq_id], shape=None, tran=raw_data["tran"][seq_id]
                )  # DIP IMU does not contain translations

                if os.path.isfile(os.path.join(MESH_DATA_PREFIX, f"{seq_id}.pt")):
                    pose_global_p, joint_p, mesh_p = torch.load(
                        os.path.join(MESH_DATA_PREFIX, f"{seq_id}.pt"),
                        weights_only=True,
                    )
                else:
                    pose_global_p, joint_p, mesh_p = model.forward_kinematics(
                        pose_p, shape_p, tran_p, calc_mesh=True
                    )
                    torch.save(
                        (pose_global_p, joint_p, mesh_p),
                        os.path.join(MESH_DATA_PREFIX, f"{seq_id}.pt"),
                    )

                seq_kwargs["glb_acc"], seq_kwargs["glb_ori"] = get_local_imu(
                    glb_acc=get_global_acc(mesh_p[:, mesh_mask], frame_rate=60),
                    glb_rot=pose_global_p[:, ji_mask],
                )

                # TODO: replace FK root_pos/root_ori with gt_pos/gt_ori

                glb_root_ori = raw_data["ori"][seq_id][:, [-1], ...]
                # glb_root_ori = pose_global_p[:, joint_set.root, ...]

                # get leaf joint position
                seq_kwargs["leaf_joint"] = get_leaf_joint(joint_p, glb_root_ori)

                # non-root joint position wrt root position
                seq_kwargs["non_root_joint"] = get_non_root_joint_pos(
                    joint_p, glb_root_ori
                )

                # get 6D joint rotation
                # ji_mask = torch.tensor([18, 19, 4, 5, 15, 0])
                # pose_global_p[:,ji_mask,...] = raw_data["ori"][seq_id] #replace globl pose with IMU data
                seq_kwargs["pose_global_6d"] = get_6d_rotation(
                    pose_global_p, glb_root_ori
                )

                # get joint velocity (for DIP joint vel label is not used)
                seq_kwargs["joint_vel"] = get_joint_vel(joint_p, glb_root_ori)

                # contact points (for DIP contact label is not used)
                seq_kwargs["feet_contact"] = get_contact_points(joint_p)

                # uwb
                uwb = raw_data["vuwb"][seq_id]
                if normalize_uwb:
                    d_root2head = uwb[
                        0, 4, 5
                    ].clone()  # normalized by distance between head and root, hard code
                    uwb = uwb / d_root2head

                if flatten_uwb:
                    index = torch.triu_indices(IMU_NUM, IMU_NUM, 1)
                    uwb = uwb.view(-1, IMU_NUM, IMU_NUM)[:, index[0], index[1]]

                if remove_node == -1:
                    seq_kwargs["uwb_m"] = uwb
                else:
                    uwb[:, remove_node, :] = 0
                    uwb[:, :, remove_node] = 0
                    seq_kwargs["uwb_m"] = uwb

                seq_kwargs["uwb_c"] = torch.zeros_like(uwb)
                seq_kwargs["uwb_gt"] = (
                    raw_data["uwb_gt"][seq_id] if "uwb_gt" in raw_data else uwb
                )
                seq_kwargs["offset"] = torch.zeros(6, 3)

                seq_kwargs["acc_sum"] = get_acc_sum(seq_kwargs["imu_acc"])

                seq_kwargs["fnames"] = raw_data["fnames"][seq_id]

                torch.save(seq_kwargs, session_file)
                print(
                    f"{self.__class__.__name__}: session {session_name} is saved at {session_file}."
                )
