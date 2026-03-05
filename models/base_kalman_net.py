# base_kalman_net.py

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class BaseKalmanNet(nn.Module, ABC):
    """
    Base class for all KalmanNet variants providing a unified interface.

    All models take noisy observations as input and produce filtered states as output,
    with internal state management for uncertainty estimates and hidden states.
    """

    def __init__(
        self,
        state_dim: int,
        obs_dim: int,
        control_dim: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.control_dim = control_dim
        self.tbptt_interval: int = 0  # 0 = full BPTT; >0 = detach every K steps

        batch_size = 1
        device = "cpu"

        # Internal state storage
        self._current_state = torch.zeros(batch_size, self.state_dim, device=device)
        self._uncertainty = (
            torch.eye(self.state_dim, device=device)
            .unsqueeze(0)
            .repeat(batch_size, 1, 1)
        )
        self._innovation = torch.zeros(batch_size, self.obs_dim, device=device)
        self._previous_obs = torch.zeros(batch_size, self.obs_dim, device=device)
        self._initialized = False

    @abstractmethod
    def _forward_step(
        self,
        observation: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        control: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Process a single observation and return the filtered state.
        This is the core method each variant must implement.

        Args:
            observation: [batch, obs_dim] - noisy observation
            mask: [batch] - optional validity mask (1=valid, 0=invalid)
            control: [batch, control_dim] - optional control input
        """

    def forward(
        self,
        observations: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        initial_state: Optional[torch.Tensor] = None,
        initial_covariance: Optional[torch.Tensor] = None,
        controls: Optional[torch.Tensor] = None,
        return_uncertainty: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Unified forward pass for all KalmanNet variants.

        Args:
            observations: [batch, seq_len, obs_dim] - noisy observations
            mask: [batch, seq_len] - optional validity mask (1=valid, 0=invalid)
            initial_state: [batch, state_dim] - optional initial state
            initial_covariance: [batch, state_dim, state_dim] - optional initial covariance
            controls: [batch, seq_len, control_dim] - optional control inputs
            return_uncertainty: whether to return uncertainty estimates

        Returns:
            result: Dictionary containing the following keys:
                states: [batch, seq_len, state_dim] - filtered state estimates
                uncertainties: [batch, seq_len, state_dim, state_dim] - if return_uncertainty=True
        """
        batch_size, seq_len, obs_dim = observations.shape
        device = observations.device

        # Initialize
        self.reset_state(
            batch_size,
            device,
            initial_state=initial_state,
            initial_covariance=initial_covariance,
        )

        # Default mask (all valid)
        if mask is None:
            mask = torch.ones(batch_size, seq_len, device=device, dtype=torch.bool)

        # Process sequence
        states = []
        uncertainties = []

        for t in range(seq_len):
            # Detach internal states to truncate the backward graph
            if self.tbptt_interval > 0 and t > 0 and t % self.tbptt_interval == 0:
                self._detach_state()

            obs_t = observations[:, t]  # [batch, obs_dim]
            mask_t = mask[:, t]  # [batch]
            u_t = controls[:, t] if controls is not None else None

            # Forward step
            state_t = self._forward_step(obs_t, mask=mask_t, control=u_t)
            states.append(state_t)

            if return_uncertainty:
                uncertainties.append(self.get_uncertainty())

        result = {"states": torch.stack(states, dim=1)}  # [batch, seq_len, state_dim]

        if return_uncertainty:
            result["uncertainties"] = torch.stack(
                uncertainties, dim=1
            )  # [batch, seq_len, state_dim, state_dim]

        return result

    def reset_state(
        self,
        batch_size: int,
        device: torch.device,
        /,
        initial_state: Optional[torch.Tensor] = None,
        initial_covariance: Optional[torch.Tensor] = None,
    ):
        """Reset all internal states."""
        if initial_state is None:
            self._current_state = torch.zeros(batch_size, self.state_dim, device=device)
        elif initial_state.shape != (batch_size, self.state_dim):
            raise ValueError(
                f"Expected shape ({batch_size}, {self.state_dim}), got {initial_state.shape}"
            )
        else:
            self._current_state = initial_state

        if initial_covariance is None:
            self._uncertainty = (
                torch.eye(self.state_dim, device=device)
                .unsqueeze(0)
                .repeat(batch_size, 1, 1)
            )
        elif initial_covariance.shape != (batch_size, self.state_dim, self.state_dim):
            raise ValueError(
                f"Expected shape ({batch_size}, {self.state_dim}, {self.state_dim}), got {initial_covariance.shape}"
            )
        else:
            self._uncertainty = initial_covariance

        self._innovation = torch.zeros(batch_size, self.obs_dim, device=device)
        self._previous_obs = torch.zeros(batch_size, self.obs_dim, device=device)
        self._initialized = False

    def _detach_state(self):
        """Detach internal states from the computation graph (truncated BPTT)."""
        if self._current_state is not None:
            self._current_state = self._current_state.detach()
        if self._uncertainty is not None:
            self._uncertainty = self._uncertainty.detach()

    def get_state(self) -> Dict[str, Any]:
        """Get current state and uncertainty estimates."""
        return {
            "state": self._current_state,
            "covariance": self._uncertainty,
            "innovation": self._innovation,
            "previous_obs": self._previous_obs,
            "initialized": self._initialized,
        }

    def get_current_state(self) -> torch.Tensor:
        """Get current state estimate."""
        return self._current_state

    def get_uncertainty(self) -> torch.Tensor:
        """Get current uncertainty estimate."""
        return self._uncertainty

    def get_innovation(self) -> torch.Tensor:
        """Get current innovation estimate."""
        return self._innovation

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Return model-specific information."""
