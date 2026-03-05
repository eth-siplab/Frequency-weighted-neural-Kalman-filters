from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from .synthetical_datasets import LorenzDataset, PendulumDataset


def visualize_lorenz_dataset(
    dataset: LorenzDataset, num_samples: int = 5, fname: Optional[str] = None
):
    """Visualize Lorenz dataset"""
    print(f"Lorenz Dataset Visualization")
    print(f"State dim: {dataset.state_dim}, Obs dim: {dataset.obs_dim}")
    print(f"Trajectories: {len(dataset)}, Sequence length: {dataset.sequence_length}")
    print(
        f"Process noise: {dataset.process_noise_std}, Measurement noise: {dataset.measurement_noise_std}"
    )
    print("-" * 60)

    fig = plt.figure(figsize=(20, 12))

    # Sample trajectories for visualization
    indices = np.random.choice(
        len(dataset), min(num_samples, len(dataset)), replace=False
    )

    # 3D Lorenz attractor plot
    ax1 = fig.add_subplot(2, 4, 1, projection="3d")
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        states = sample["states"].numpy()
        ax1.plot(states[:, 0], states[:, 1], states[:, 2], alpha=0.7, linewidth=1.5)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_title("Lorenz Attractor (3D)")

    # Phase space projections
    ax2 = fig.add_subplot(2, 4, 2)
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        states = sample["states"].numpy()
        ax2.plot(states[:, 0], states[:, 1], alpha=0.7, linewidth=1.5)
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_title("X-Y Phase Plane")
    ax2.grid(True)

    # Time series - single trajectory
    sample = dataset[indices[0]]
    states = sample["states"].numpy()
    observations = sample["observations"].numpy()
    time = np.arange(len(states)) * dataset.dt

    # X coordinate
    ax3 = fig.add_subplot(2, 4, 3)
    ax3.plot(time, states[:, 0], "b-", label="True X", linewidth=2)
    if dataset.obs_dim >= 1:
        ax3.plot(
            time, observations[:, 0], "r.", label="Observed X", alpha=0.6, markersize=3
        )
    ax3.set_xlabel("Time")
    ax3.set_ylabel("X")
    ax3.set_title("X Coordinate Time Series")
    ax3.legend()
    ax3.grid(True)

    # Y coordinate
    ax4 = fig.add_subplot(2, 4, 4)
    ax4.plot(time, states[:, 1], "b-", label="True Y", linewidth=2)
    if dataset.obs_dim >= 2:
        ax4.plot(
            time, observations[:, 1], "r.", label="Observed Y", alpha=0.6, markersize=3
        )
    ax4.set_xlabel("Time")
    ax4.set_ylabel("Y")
    ax4.set_title("Y Coordinate Time Series")
    ax4.legend()
    ax4.grid(True)

    # Z coordinate
    ax5 = fig.add_subplot(2, 4, 5)
    ax5.plot(time, states[:, 2], "b-", label="True Z", linewidth=2)
    if dataset.obs_dim >= 3:
        ax5.plot(
            time, observations[:, 2], "r.", label="Observed Z", alpha=0.6, markersize=3
        )
    ax5.set_xlabel("Time")
    ax5.set_ylabel("Z")
    ax5.set_title("Z Coordinate Time Series")
    ax5.legend()
    ax5.grid(True)

    # Observation vs True scatter plots
    ax6 = fig.add_subplot(2, 4, 6)
    if dataset.obs_dim >= 1:
        ax6.scatter(states[:, 0], observations[:, 0], alpha=0.5, s=10)
        min_val, max_val = min(states[:, 0].min(), observations[:, 0].min()), max(
            states[:, 0].max(), observations[:, 0].max()
        )
        ax6.plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.8)
    ax6.set_xlabel("True X")
    ax6.set_ylabel("Observed X")
    ax6.set_title("True vs Observed X")
    ax6.grid(True)

    # Distribution of noise
    ax7 = fig.add_subplot(2, 4, 7)
    if dataset.obs_dim >= 1:
        noise_x = observations[:, 0] - states[:, 0]
        ax7.hist(noise_x, bins=30, alpha=0.7, density=True)
        ax7.axvline(
            noise_x.mean(),
            color="red",
            linestyle="--",
            label=f"Mean: {noise_x.mean():.3f}",
        )
        ax7.axvline(
            noise_x.std(),
            color="orange",
            linestyle="--",
            label=f"Std: {noise_x.std():.3f}",
        )
    ax7.set_xlabel("Measurement Noise (X)")
    ax7.set_ylabel("Density")
    ax7.set_title("Measurement Noise Distribution")
    ax7.legend()
    ax7.grid(True)

    # Multiple trajectories overlay
    ax8 = fig.add_subplot(2, 4, 8)
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        states = sample["states"].numpy()
        ax8.plot(
            states[:, 0], states[:, 2], alpha=0.6, linewidth=1.5, label=f"Traj {i+1}"
        )
    ax8.set_xlabel("X")
    ax8.set_ylabel("Z")
    ax8.set_title("X-Z Phase Plane (Multiple Trajectories)")
    ax8.legend()
    ax8.grid(True)

    plt.tight_layout()
    if fname:
        plt.savefig(fname)
    else:
        plt.show()


def visualize_pendulum_dataset(
    dataset: PendulumDataset, num_samples: int = 5, fname: Optional[str] = None
):
    """Visualize Pendulum dataset"""
    print(f"\nPendulum Dataset Visualization")
    print(f"State dim: {dataset.state_dim}, Obs dim: {dataset.obs_dim}")
    print(f"Trajectories: {len(dataset)}, Sequence length: {dataset.sequence_length}")
    print(
        f"Process noise: {dataset.process_noise_std}, Measurement noise: {dataset.measurement_noise_std}"
    )
    print(f"Length: {dataset.L}m, Damping: {dataset.b}, Mass: {dataset.m}kg")
    print("-" * 60)

    fig = plt.figure(figsize=(20, 12))

    # Sample trajectories for visualization
    indices = np.random.choice(
        len(dataset), min(num_samples, len(dataset)), replace=False
    )

    # Pendulum animation frames
    ax1 = fig.add_subplot(2, 4, 1)
    sample = dataset[indices[0]]
    states = sample["states"].numpy()
    observations = sample["observations"].numpy()

    # Plot pendulum positions at different times
    time_steps = np.linspace(0, len(states) - 1, 20, dtype=int)
    for i, t in enumerate(time_steps):
        theta = states[t, 0]
        x = dataset.L * np.sin(theta)
        y = -dataset.L * np.cos(theta)

        # Pendulum rod
        ax1.plot(
            [0, x], [0, y], "b-", alpha=0.3 + 0.7 * i / len(time_steps), linewidth=2
        )
        # Pendulum bob
        ax1.plot(x, y, "ro", markersize=4, alpha=0.3 + 0.7 * i / len(time_steps))

    ax1.set_xlim(-1.5 * dataset.L, 1.5 * dataset.L)
    ax1.set_ylim(-1.5 * dataset.L, 0.5 * dataset.L)
    ax1.set_aspect("equal")
    ax1.set_xlabel("X Position")
    ax1.set_ylabel("Y Position")
    ax1.set_title("Pendulum Motion (Overlay)")
    ax1.grid(True)

    # Phase portrait (θ vs ω)
    ax2 = fig.add_subplot(2, 4, 2)
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        states = sample["states"].numpy()
        ax2.plot(states[:, 0], states[:, 1], alpha=0.7, linewidth=1.5)
    ax2.set_xlabel("Angle θ (rad)")
    ax2.set_ylabel("Angular Velocity ω (rad/s)")
    ax2.set_title("Phase Portrait (θ vs ω)")
    ax2.grid(True)

    # Time series - angle
    time = np.arange(len(states)) * dataset.dt
    ax3 = fig.add_subplot(2, 4, 3)
    ax3.plot(time, states[:, 0], "b-", label="True θ", linewidth=2)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Angle θ (rad)")
    ax3.set_title("Angle Time Series")
    ax3.legend()
    ax3.grid(True)

    # Time series - angular velocity
    ax4 = fig.add_subplot(2, 4, 4)
    ax4.plot(time, states[:, 1], "g-", label="True ω", linewidth=2)
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Angular Velocity ω (rad/s)")
    ax4.set_title("Angular Velocity Time Series")
    ax4.legend()
    ax4.grid(True)

    # Cartesian observations
    ax5 = fig.add_subplot(2, 4, 5)
    # True Cartesian coordinates
    true_x = dataset.L * np.sin(states[:, 0])
    true_y = -dataset.L * np.cos(states[:, 0])
    ax5.plot(true_x, true_y, "b-", label="True Position", linewidth=2)
    ax5.plot(
        observations[:, 0],
        observations[:, 1],
        "r.",
        label="Observed Position",
        alpha=0.6,
        markersize=3,
    )
    ax5.set_xlabel("X Position")
    ax5.set_ylabel("Y Position")
    ax5.set_title("Cartesian Position (True vs Observed)")
    ax5.legend()
    ax5.grid(True)
    ax5.set_aspect("equal")

    # Control input
    ax6 = fig.add_subplot(2, 4, 6)
    controls = sample["controls"].numpy()
    ax6.plot(time, controls, "purple", alpha=0.7, linewidth=1.5)
    ax6.set_xlabel("Time (s)")
    ax6.set_ylabel("Control Input")
    ax6.set_title("Control Input Time Series")
    ax6.grid(True)

    # Observation error
    ax7 = fig.add_subplot(2, 4, 7)
    error_x = observations[:, 0] - true_x
    error_y = observations[:, 1] - true_y
    ax7.plot(time, error_x, label="X Error", alpha=0.7)
    ax7.plot(time, error_y, label="Y Error", alpha=0.7)
    ax7.set_xlabel("Time (s)")
    ax7.set_ylabel("Observation Error")
    ax7.set_title("Observation Error Time Series")
    ax7.legend()
    ax7.grid(True)

    # Energy analysis
    ax8 = fig.add_subplot(2, 4, 8)
    # Kinetic energy: 0.5 * m * L^2 * ω^2
    kinetic_energy = 0.5 * dataset.m * (dataset.L * states[:, 1]) ** 2
    # Potential energy: m * g * L * (1 - cos(θ))
    potential_energy = dataset.m * dataset.g * dataset.L * (1 - np.cos(states[:, 0]))
    total_energy = kinetic_energy + potential_energy

    ax8.plot(time, kinetic_energy, label="Kinetic", alpha=0.7)
    ax8.plot(time, potential_energy, label="Potential", alpha=0.7)
    ax8.plot(time, total_energy, label="Total", alpha=0.7, linewidth=2)
    ax8.set_xlabel("Time (s)")
    ax8.set_ylabel("Energy (J)")
    ax8.set_title("Energy Components")
    ax8.legend()
    ax8.grid(True)

    plt.tight_layout()
    if fname:
        plt.savefig(fname)
    else:
        plt.show()


def compare_datasets(fname: Optional[str] = None):
    """Compare characteristics of both datasets"""
    print("\n" + "=" * 80)
    print("DATASET COMPARISON")
    print("=" * 80)

    # Create small datasets for comparison
    lorenz_dataset = LorenzDataset(num_trajectories=100, sequence_length=200)
    pendulum_dataset = PendulumDataset(num_trajectories=100, sequence_length=200)

    datasets = {"Lorenz": lorenz_dataset, "Pendulum": pendulum_dataset}

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for i, (name, dataset) in enumerate(datasets.items()):
        # Get a sample trajectory
        sample = dataset[0]
        states = sample["states"].numpy()
        observations = sample["observations"].numpy()

        # State space dimensionality
        ax = axes[i, 0]
        if name == "Lorenz":
            ax.plot(
                states[:, 0], states[:, 1], "b-", linewidth=1.5, label="True trajectory"
            )
            ax.scatter(
                observations[:, 0],
                observations[:, 1],
                c="red",
                s=10,
                alpha=0.6,
                label="Observations",
            )
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_title(f"{name}: State Space (X-Y plane)")
        else:
            ax.plot(
                states[:, 0], states[:, 1], "b-", linewidth=1.5, label="True trajectory"
            )
            ax.set_xlabel("Angle θ (rad)")
            ax.set_ylabel("Angular velocity ω (rad/s)")
            ax.set_title(f"{name}: Phase Space (θ-ω plane)")
        ax.legend()
        ax.grid(True)

        # Temporal dynamics
        ax = axes[i, 1]
        time = np.arange(len(states)) * (0.01 if name == "Lorenz" else 0.02)
        ax.plot(time, states[:, 0], label="State 1", linewidth=1.5)
        ax.plot(time, states[:, 1], label="State 2", linewidth=1.5)
        if states.shape[1] > 2:
            ax.plot(time, states[:, 2], label="State 3", linewidth=1.5)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("State Value")
        ax.set_title(f"{name}: Temporal Evolution")
        ax.legend()
        ax.grid(True)

        # Observation characteristics
        ax = axes[i, 2]
        ax.plot(time, observations[:, 0], "r-", label="Obs 1", alpha=0.8, linewidth=1.5)
        ax.plot(time, observations[:, 1], "g-", label="Obs 2", alpha=0.8, linewidth=1.5)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Observation Value")
        ax.set_title(f"{name}: Observations")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    if fname:
        plt.savefig(fname)
    else:
        plt.show()

    # Print statistics
    print(
        f"{'Dataset':<12} {'State Dim':<10} {'Obs Dim':<10} {'Dynamics':<15} {'Linearity':<12}"
    )
    print("-" * 70)
    print(
        f"{'Lorenz':<12} {lorenz_dataset.state_dim:<10} {lorenz_dataset.obs_dim:<10} {'Chaotic':<15} {'Nonlinear':<12}"
    )
    print(
        f"{'Pendulum':<12} {pendulum_dataset.state_dim:<10} {pendulum_dataset.obs_dim:<10} {'Oscillatory':<15} {'Nonlinear':<12}"
    )

    # Compute some statistics
    for name, dataset in datasets.items():
        states_all = dataset.states.numpy()  # [num_traj, seq_len, state_dim]
        obs_all = dataset.observations.numpy()  # [num_traj, seq_len, obs_dim]

        print(f"\n{name} Dataset Statistics:")
        print(
            f"  State ranges: {states_all.min(axis=(0,1))} to {states_all.max(axis=(0,1))}"
        )
        print(f"  State std: {states_all.std(axis=(0,1))}")
        print(
            f"  Observation ranges: {obs_all.min(axis=(0,1))} to {obs_all.max(axis=(0,1))}"
        )
        print(f"  Observation std: {obs_all.std(axis=(0,1))}")


def demo_dataloader_usage():
    """Demonstrate how to use datasets with PyTorch DataLoader"""
    print("\n" + "=" * 80)
    print("DATALOADER DEMO")
    print("=" * 80)

    # Create datasets
    lorenz_dataset = LorenzDataset(num_trajectories=32, sequence_length=50)
    pendulum_dataset = PendulumDataset(num_trajectories=32, sequence_length=50)

    # Create dataloaders
    lorenz_loader = torch.utils.data.DataLoader(
        lorenz_dataset, batch_size=8, shuffle=True
    )
    pendulum_loader = torch.utils.data.DataLoader(
        pendulum_dataset, batch_size=8, shuffle=True
    )

    print("DataLoader Examples:")
    print("-" * 30)

    # Test Lorenz dataloader
    for i, batch in enumerate(lorenz_loader):
        print(f"Lorenz Batch {i+1}:")
        print(f"  States shape: {batch['states'].shape}")
        print(f"  Observations shape: {batch['observations'].shape}")
        if i == 0:  # Only show first batch
            break

    # Test Pendulum dataloader
    for i, batch in enumerate(pendulum_loader):
        print(f"\nPendulum Batch {i+1}:")
        print(f"  States shape: {batch['states'].shape}")
        print(f"  Observations shape: {batch['observations'].shape}")
        print(f"  Controls shape: {batch['controls'].shape}")
        if i == 0:  # Only show first batch
            break


if __name__ == "__main__":
    """
    Run this script to visualize both datasets:
    python loader.py
    """
    print("KalmanNet Dataset Visualization")
    print("=" * 80)

    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Create datasets
    print("Creating datasets...")
    # Lorenz — under-sampled, noisier, more chaotic
    lorenz_dataset = LorenzDataset(
        num_trajectories=500,
        sequence_length=120,  # less context
        dt=0.02,  # under-sample → aliasing
        sigma=10.0,
        rho=35.0,
        beta=8 / 3,  # stronger chaos than ρ=28
        process_noise_std=0.30,  # heavier process noise
        measurement_noise_std=1.00,  # lower SNR
        partial_obs=True,  # keep it partial
    )

    # Pendulum — weak damping + strong forcing + lower SNR
    pendulum_dataset = PendulumDataset(
        num_trajectories=500,
        sequence_length=120,  # less context
        dt=0.04,  # under-sample dynamics
        L=0.4,  # smaller amplitude → lower SNR (x,y shrink)
        b=0.02,  # low damping → longer transients
        control_amplitude=1.2,  # kick it around
        process_noise_std=0.12,
        measurement_noise_std=0.35,
    )

    # Visualize Lorenz dataset
    visualize_lorenz_dataset(lorenz_dataset, num_samples=3, fname="lorenz_dataset.png")

    # Visualize Pendulum dataset
    visualize_pendulum_dataset(
        pendulum_dataset, num_samples=3, fname="pendulum_dataset.png"
    )

    # Compare datasets
    compare_datasets(fname="dataset_comparison.png")

    # Demo dataloader usage
    demo_dataloader_usage()

    print("\nVisualization complete!")
    print("Datasets are ready for KalmanNet training.")
