import torch
from typing import Optional
from src.utils.rigid_utils import Rigids


def compute_fape_squared(
        pred_frames: Rigids,
        target_frames: Rigids,
        frames_mask: torch.Tensor,
        pred_positions: torch.Tensor,
        target_positions: torch.Tensor,
        positions_mask: torch.Tensor,
        length_scale: float = 10.0,
        l2_clamp_distance: Optional[float] = None,
        eps=1e-8,
) -> torch.Tensor:
    """
        Computes squared FAPE loss.

        Args:
            pred_frames:
                [*, N_frames] Rigid object of predicted frames
            target_frames:
                [*, N_frames] Rigid object of ground truth frames
            frames_mask:
                [*, N_frames] binary mask for the frames
            pred_positions:
                [*, N_pts, 3] predicted atom positions
            target_positions:
                [*, N_pts, 3] ground truth positions
            positions_mask:
                [*, N_pts] positions mask
            length_scale:
                Length scale by which the loss is divided
            l2_clamp_distance:
                Cutoff above which squared distance errors are disregarded.
            eps:
                Small value used to regularize denominators
        Returns:
            [*] loss tensor
    """
    # [*, N_frames, N_pts, 3]
    local_pred_pos = pred_frames.invert()[..., None].apply(
        pred_positions[..., None, :, :],
    )
    local_target_pos = target_frames.invert()[..., None].apply(
        target_positions[..., None, :, :],
    )

    error_dist = torch.sum((local_pred_pos - local_target_pos) ** 2, dim=-1)

    if l2_clamp_distance is not None:
        error_dist = torch.clamp(error_dist, min=0, max=l2_clamp_distance)

    normed_error = error_dist / length_scale
    normed_error = normed_error * frames_mask[..., None]
    normed_error = normed_error * positions_mask[..., None, :]

    # FP16-friendly averaging. Roughly equivalent to:
    #
    # norm_factor = (
    #     torch.sum(frames_mask, dim=-1) *
    #     torch.sum(positions_mask, dim=-1)
    # )
    # normed_error = torch.sum(normed_error, dim=(-1, -2)) / (eps + norm_factor)
    #
    # ("roughly" because eps is necessarily duplicated in the latter)
    normed_error = torch.sum(normed_error, dim=-1)
    normed_error = (
            normed_error / (eps + torch.sum(frames_mask, dim=-1))[..., None]
    )
    normed_error = torch.sum(normed_error, dim=-1)
    normed_error = normed_error / (eps + torch.sum(positions_mask, dim=-1))

    return normed_error


def fape_squared_with_clamp(
        pred_frames: Rigids,
        target_frames: Rigids,
        frames_mask: torch.Tensor,
        pred_positions: torch.Tensor,
        target_positions: torch.Tensor,
        positions_mask: torch.Tensor,
        use_clamped_fape: float = 0.9,
        l2_clamp_distance: float = 100.0,  # 10A ^ 2
        eps: float = 1e-4,
        **kwargs,
) -> torch.Tensor:
    """Compute squared FAPE loss with clamping.
        Args:
                pred_frames:
                    [*, N_frames] Rigid object of predicted frames
                target_frames:
                    [*, N_frames] Rigid object of ground truth frames
                frames_mask:
                    [*, N_frames] binary mask for the frames
                pred_positions:
                    [*, N_pts, 3] predicted atom positions
                target_positions:
                    [*, N_pts, 3] ground truth positions
                positions_mask:
                    [*, N_pts] positions mask
                use_clamped_fape:
                    ratio of clamped to unclamped FAPE in final loss
                l2_clamp_distance:
                    Cutoff above which squared distance errors are disregarded.
                eps:
                    Small value used to regularize denominators
            Returns:
                [*] loss tensor
    """
    fape_loss = compute_fape_squared(pred_frames=pred_frames,
                                     target_frames=target_frames,
                                     frames_mask=frames_mask,
                                     pred_positions=pred_positions,
                                     target_positions=target_positions,
                                     positions_mask=positions_mask,
                                     l2_clamp_distance=l2_clamp_distance,
                                     eps=eps)
    if use_clamped_fape is not None:
        unclamped_fape_loss = compute_fape_squared(pred_frames=pred_frames,
                                                   target_frames=target_frames,
                                                   frames_mask=frames_mask,
                                                   pred_positions=pred_positions,
                                                   target_positions=target_positions,
                                                   positions_mask=positions_mask,
                                                   l2_clamp_distance=l2_clamp_distance,
                                                   eps=eps)
        use_clamped_fape = torch.Tensor([use_clamped_fape]).cuda()  # for proper multiplication
        # Average the two to provide a useful training signal even early on in training.
        fape_loss = fape_loss * use_clamped_fape + unclamped_fape_loss * (
                1 - use_clamped_fape
        )

    # Average over the batch dimension
    fape_loss = torch.mean(fape_loss)

    return fape_loss