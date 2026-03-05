import torch
import torch.nn.functional as F


def rotation_matrix_to_quaternion(rotation_matrices: torch.Tensor) -> torch.Tensor:
    """
    Convert batched 3x3 rotation matrices to quaternions.

    Args:
        rotation_matrices: Tensor of shape [..., 3, 3] containing rotation matrices

    Returns:
        quaternions: Tensor of shape [..., 4] containing quaternions in (w, x, y, z) format
    """
    # Get the shape for later reshaping
    original_shape = rotation_matrices.shape[:-2]

    # Flatten all batch dimensions into a single dimension
    matrices = rotation_matrices.view(-1, 3, 3)
    batch_size = matrices.shape[0]

    # Extract matrix elements for readability
    m00, m01, m02 = matrices[:, 0, 0], matrices[:, 0, 1], matrices[:, 0, 2]
    m10, m11, m12 = matrices[:, 1, 0], matrices[:, 1, 1], matrices[:, 1, 2]
    m20, m21, m22 = matrices[:, 2, 0], matrices[:, 2, 1], matrices[:, 2, 2]

    # Compute the trace
    trace = m00 + m11 + m22

    # Initialize quaternion tensor
    quaternions = torch.zeros(
        batch_size, 4, device=matrices.device, dtype=matrices.dtype
    )

    # Case 1: trace > 0
    trace_positive_mask = trace > 0
    if torch.any(trace_positive_mask):
        s = torch.sqrt(trace[trace_positive_mask] + 1.0) * 2  # s = 4 * qw
        quaternions[trace_positive_mask, 0] = 0.25 * s  # qw
        quaternions[trace_positive_mask, 1] = (
            m21[trace_positive_mask] - m12[trace_positive_mask]
        ) / s  # qx
        quaternions[trace_positive_mask, 2] = (
            m02[trace_positive_mask] - m20[trace_positive_mask]
        ) / s  # qy
        quaternions[trace_positive_mask, 3] = (
            m10[trace_positive_mask] - m01[trace_positive_mask]
        ) / s  # qz

    # Case 2: m00 is the largest diagonal element
    remaining_mask = ~trace_positive_mask
    m00_largest_mask = remaining_mask & (m00 > m11) & (m00 > m22)
    if torch.any(m00_largest_mask):
        s = (
            torch.sqrt(
                1.0
                + m00[m00_largest_mask]
                - m11[m00_largest_mask]
                - m22[m00_largest_mask]
            )
            * 2
        )  # s = 4 * qx
        quaternions[m00_largest_mask, 0] = (
            m21[m00_largest_mask] - m12[m00_largest_mask]
        ) / s  # qw
        quaternions[m00_largest_mask, 1] = 0.25 * s  # qx
        quaternions[m00_largest_mask, 2] = (
            m01[m00_largest_mask] + m10[m00_largest_mask]
        ) / s  # qy
        quaternions[m00_largest_mask, 3] = (
            m02[m00_largest_mask] + m20[m00_largest_mask]
        ) / s  # qz

    # Case 3: m11 is the largest diagonal element
    remaining_mask = remaining_mask & ~m00_largest_mask
    m11_largest_mask = remaining_mask & (m11 > m22)
    if torch.any(m11_largest_mask):
        s = (
            torch.sqrt(
                1.0
                + m11[m11_largest_mask]
                - m00[m11_largest_mask]
                - m22[m11_largest_mask]
            )
            * 2
        )  # s = 4 * qy
        quaternions[m11_largest_mask, 0] = (
            m02[m11_largest_mask] - m20[m11_largest_mask]
        ) / s  # qw
        quaternions[m11_largest_mask, 1] = (
            m01[m11_largest_mask] + m10[m11_largest_mask]
        ) / s  # qx
        quaternions[m11_largest_mask, 2] = 0.25 * s  # qy
        quaternions[m11_largest_mask, 3] = (
            m12[m11_largest_mask] + m21[m11_largest_mask]
        ) / s  # qz

    # Case 4: m22 is the largest diagonal element
    remaining_mask = remaining_mask & ~m11_largest_mask
    if torch.any(remaining_mask):
        s = (
            torch.sqrt(
                1.0 + m22[remaining_mask] - m00[remaining_mask] - m11[remaining_mask]
            )
            * 2
        )  # s = 4 * qz
        quaternions[remaining_mask, 0] = (
            m10[remaining_mask] - m01[remaining_mask]
        ) / s  # qw
        quaternions[remaining_mask, 1] = (
            m02[remaining_mask] + m20[remaining_mask]
        ) / s  # qx
        quaternions[remaining_mask, 2] = (
            m12[remaining_mask] + m21[remaining_mask]
        ) / s  # qy
        quaternions[remaining_mask, 3] = 0.25 * s  # qz

    # Normalize quaternions to ensure they're unit quaternions
    quaternions = F.normalize(quaternions, p=2, dim=1)

    # Ensure positive scalar part (w) by convention
    negative_w_mask = quaternions[:, 0] < 0
    quaternions[negative_w_mask] *= -1

    # Reshape back to original batch dimensions + quaternion dimension
    quaternions = quaternions.view(*original_shape, 4)

    return quaternions
