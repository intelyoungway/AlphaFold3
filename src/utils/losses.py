import torch
from typing import Dict, Optional

from src.common import residue_constants
from src.utils.rigid_utils import Rotations, Rigids
from src.utils.geometry.rigid_matrix_vector import Rigid3Array
from src.utils.geometry import vector
from src.utils.geometry.vector import Vec3Array
from src.utils.tensor_utils import (
    masked_mean,
    permute_final_dims,
)


def softmax_cross_entropy(logits, labels):
    loss = -1 * torch.sum(
        labels * torch.nn.functional.log_softmax(logits, dim=-1),
        dim=-1,
    )
    return loss


def sigmoid_cross_entropy(logits, labels):
    logits_dtype = logits.dtype
    logits = logits.double()
    labels = labels.double()
    log_p = torch.nn.functional.logsigmoid(logits)
    log_not_p = torch.nn.functional.logsigmoid(-1 * logits)
    loss = (-1. * labels) * log_p - (1. - labels) * log_not_p
    loss = loss.to(dtype=logits_dtype)
    return loss


def frame_aligned_point_error(
        pred_frames: Rigid3Array,  # shape [*, num_frames]
        target_frames: Rigid3Array,  # shape [*, num_frames]
        frames_mask: torch.Tensor,  # shape [*, num_frames]
        pred_positions: Vec3Array,  # shape [*, num_positions]
        target_positions: Vec3Array,  # shape [*, num_positions]
        positions_mask: torch.Tensor,  # shape [*, num_positions]
        length_scale: float = 10.0,
        l1_clamp_distance: Optional[float] = None,
        clamp_fraction: Optional[float] = 0.90,
        epsilon=1e-8,
        squared: bool = False,
) -> torch.Tensor:  # shape [*]
    """Measure point error under different alignments.

    Jumper et al. (2021) Suppl. Alg. 28 "computeFAPE"

    Computes error between two structures with B points under A alignments derived
    from the given pairs of frames.
    Args:
        pred_frames:
            num_frames reference frames for 'pred_positions'.
        target_frames:
            num_frames reference frames for 'target_positions'.
        frames_mask:
            Mask for frame pairs to use.
        pred_positions:
            num_positions predicted positions of the structure.
        target_positions:
            num_positions target positions of the structure.
        positions_mask:
            Mask on which positions to score.
        length_scale:
            length scale to divide loss by.
        l1_clamp_distance:
            Distance cutoff on error beyond which gradients will be zero.
        clamp_fraction:
            the fraction of examples to clamp in the batch
        epsilon:
            small value used to regularize denominator for masked average.
        squared:
            If True, return squared error.
    Returns:
        Masked Frame Aligned Point Error.
    """
    # Compute array of predicted positions in the predicted frames.
    # Vec3Array (*, num_frames, num_positions)
    local_pred_pos = pred_frames[..., None].apply_inverse_to_point(pred_positions[:, None, ...])

    # Compute array of target positions in the target frames.
    # Vec3Array (*, num_frames, num_positions)
    local_target_pos = target_frames[..., None].apply_inverse_to_point(target_positions[:, None, ...])

    # Compute errors between the structures.
    # torch.Tensor (*, num_frames, num_positions)
    error_dist = vector.square_euclidean_distance(local_pred_pos, local_target_pos)
    if not squared:
        error_dist = torch.sqrt(error_dist + epsilon)

    if l1_clamp_distance is not None:
        clamp_distance = l1_clamp_distance
        if squared:  # convert to l2 clamped distance if squared
            clamp_distance = clamp_distance ** 2

        # Determine the number of examples to clamp
        batch_size = pred_frames.shape[0]
        num_to_clamp = int(batch_size * clamp_fraction)

        # Clamp a fraction of the batch
        clamped_top_fraction = torch.clamp(error_dist[:num_to_clamp], min=0, max=clamp_distance)
        error_dist = torch.cat((clamped_top_fraction, error_dist[num_to_clamp:]), dim=0)

    if squared:
        length_scale = length_scale ** 2  # squared length scale to make loss unitless

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
            normed_error / (epsilon + torch.sum(frames_mask, dim=-1))[..., None]
    )
    normed_error = torch.sum(normed_error, dim=-1)
    normed_error = normed_error / (epsilon + torch.sum(positions_mask, dim=-1))

    return torch.mean(normed_error)  # average over batch dim


def lddt(
        all_atom_pred_pos: torch.Tensor,
        all_atom_positions: torch.Tensor,
        all_atom_mask: torch.Tensor,
        cutoff: float = 15.0,
        eps: float = 1e-10,
        per_residue: bool = True,
) -> torch.Tensor:
    """Compute the local distance difference test (LDDT) score.
    TODO: there is something off with per_residue=True, it seems to be the other way around
    TODO: the lddt values seem far too high as I noise the protein, it doesn'timesteps go below 0.8.
    """
    n = all_atom_mask.shape[-2]
    dmat_true = torch.sqrt(
        eps
        + torch.sum(
            (
                    all_atom_positions[..., None, :]
                    - all_atom_positions[..., None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    dmat_pred = torch.sqrt(
        eps
        + torch.sum(
            (
                    all_atom_pred_pos[..., None, :]
                    - all_atom_pred_pos[..., None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )
    dists_to_score = (
            (dmat_true < cutoff)
            * all_atom_mask
            * permute_final_dims(all_atom_mask, (1, 0))
            * (1.0 - torch.eye(n, device=all_atom_mask.device))
    )

    dist_l1 = torch.abs(dmat_true - dmat_pred)

    score = (
            (dist_l1 < 0.5).type(dist_l1.dtype)
            + (dist_l1 < 1.0).type(dist_l1.dtype)
            + (dist_l1 < 2.0).type(dist_l1.dtype)
            + (dist_l1 < 4.0).type(dist_l1.dtype)
    )
    score = score * 0.25

    dims = (-1,) if per_residue else (-2, -1)
    norm = 1.0 / (eps + torch.sum(dists_to_score, dim=dims))
    score = norm * (eps + torch.sum(dists_to_score * score, dim=dims))

    return score


def lddt_ca(
        all_atom_pred_pos: torch.Tensor,
        all_atom_positions: torch.Tensor,
        all_atom_mask: torch.Tensor,
        cutoff: float = 15.0,
        eps: float = 1e-10,
        per_residue: bool = True,
) -> torch.Tensor:
    """Compute the local distance difference test (LDDT) score for C-alpha atoms."""
    ca_pos = residue_constants.atom_order["CA"]
    all_atom_pred_pos = all_atom_pred_pos[..., ca_pos, :]
    all_atom_positions = all_atom_positions[..., ca_pos, :]
    all_atom_mask = all_atom_mask[..., ca_pos: (ca_pos + 1)]  # keep dim

    return lddt(
        all_atom_pred_pos,
        all_atom_positions,
        all_atom_mask,
        cutoff=cutoff,
        eps=eps,
        per_residue=per_residue,
    )


def between_residue_clash_loss(
        atom14_pred_positions: torch.Tensor,
        atom14_atom_exists: torch.Tensor,
        atom14_atom_radius: torch.Tensor,
        residue_index: torch.Tensor,
        overlap_tolerance_soft=1.5,
        overlap_tolerance_hard=1.5,
        eps=1e-10,
) -> Dict[str, torch.Tensor]:
    """Loss to penalize steric clashes between residues.

    This is a loss penalizing any steric clashes due to non-bonded atoms in
    different peptides coming too close. This loss corresponds to the part with
    different residues of
    Jumper et al. (2021) Suppl. Sec. 1.9.11, eq 46.

    Args:
      atom14_pred_positions: Predicted positions of atoms in
        global prediction frame
      atom14_atom_exists: Mask denoting whether atom at positions exists for given
        amino acid type
      atom14_atom_radius: Van der Waals radius for each atom.
      residue_index: Residue index for given amino acid.
      overlap_tolerance_soft: Soft tolerance factor.
      overlap_tolerance_hard: Hard tolerance factor.
      eps: epsilon for numerical stability.

    Returns:
      Dict containing:
        * 'mean_loss': average clash loss
        * 'per_atom_loss_sum': sum of all clash losses per atom, shape (N, 14)
        * 'per_atom_clash_mask': mask whether atom clashes with any other atom
            shape (N, 14)
    """
    fp_type = atom14_pred_positions.dtype

    # Create the distance matrix.
    # (N, N, 14, 14)
    dists = torch.sqrt(
        eps
        + torch.sum(
            (
                    atom14_pred_positions[..., :, None, :, None, :]
                    - atom14_pred_positions[..., None, :, None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    # Create the mask for valid distances.
    # shape (N, N, 14, 14)
    dists_mask = (
            atom14_atom_exists[..., :, None, :, None]
            * atom14_atom_exists[..., None, :, None, :]
    ).type(fp_type)

    # Mask out all the duplicate entries in the lower triangular matrix.
    # Also mask out the diagonal (atom-pairs from the same residue) -- these atoms
    # are handled separately.
    dists_mask = dists_mask * (
            residue_index[..., :, None, None, None]
            < residue_index[..., None, :, None, None]
    )

    # Backbone C--N bond between subsequent residues is no clash.
    c_one_hot = torch.nn.functional.one_hot(
        residue_index.new_tensor(2), num_classes=14
    )
    c_one_hot = c_one_hot.reshape(
        *((1,) * len(residue_index.shape[:-1])), *c_one_hot.shape
    )
    c_one_hot = c_one_hot.type(fp_type)
    n_one_hot = torch.nn.functional.one_hot(
        residue_index.new_tensor(0), num_classes=14
    )
    n_one_hot = n_one_hot.reshape(
        *((1,) * len(residue_index.shape[:-1])), *n_one_hot.shape
    )
    n_one_hot = n_one_hot.type(fp_type)

    neighbour_mask = (
                             residue_index[..., :, None, None, None] + 1
                     ) == residue_index[..., None, :, None, None]
    c_n_bonds = (
            neighbour_mask
            * c_one_hot[..., None, None, :, None]
            * n_one_hot[..., None, None, None, :]
    )
    dists_mask = dists_mask * (1.0 - c_n_bonds)

    # DISCREPANCY: The disulfide bridges are not taken into account when computing clashes.
    # Disulfide bridge between two cysteines is no clash.
    # cys = residue_constants.restype_name_to_atom14_names["CYS"]
    # cys_sg_idx = cys.index("SG")
    # cys_sg_idx = residue_index.new_tensor(cys_sg_idx)
    # cys_sg_idx = cys_sg_idx.reshape(
    #    *((1,) * len(residue_index.shape[:-1])), 1
    # ).squeeze(-1)
    # cys_sg_one_hot = torch.nn.functional.one_hot(cys_sg_idx, num_classes=14)
    # disulfide_bonds = (
    #         cys_sg_one_hot[..., None, None, :, None]
    #         * cys_sg_one_hot[..., None, None, None, :]
    # )
    # dists_mask = dists_mask * (1.0 - disulfide_bonds)

    # Compute the lower bound for the allowed distances.
    # shape (N, N, 14, 14)
    dists_lower_bound = dists_mask * (
            atom14_atom_radius[..., :, None, :, None]
            + atom14_atom_radius[..., None, :, None, :]
    )

    # Compute the error.
    # shape (N, N, 14, 14)
    dists_to_low_error = dists_mask * torch.nn.functional.relu(
        dists_lower_bound - overlap_tolerance_soft - dists
    )

    # Compute the mean loss.
    # shape ()
    mean_loss = torch.sum(dists_to_low_error) / (1e-6 + torch.sum(dists_mask))

    # Compute the per atom loss sum.
    # shape (N, 14)
    per_atom_loss_sum = torch.sum(dists_to_low_error, dim=(-4, -2)) + torch.sum(
        dists_to_low_error, dim=(-3, -1)
    )

    # Compute the hard clash mask.
    # shape (N, N, 14, 14)
    clash_mask = dists_mask * (
            dists < (dists_lower_bound - overlap_tolerance_hard)
    )

    # Compute the per atom clash.
    # shape (N, 14)
    per_atom_clash_mask = torch.maximum(
        torch.amax(clash_mask, dim=(-4, -2)),
        torch.amax(clash_mask, dim=(-3, -1)),
    )

    return {
        "mean_loss": mean_loss,  # shape ()
        "per_atom_loss_sum": per_atom_loss_sum,  # shape (N, 14)
        "per_atom_clash_mask": per_atom_clash_mask,  # shape (N, 14)
    }


def extreme_ca_ca_distance_violations(
        pred_atom_positions: torch.Tensor,  # (N, 37(14), 3)
        pred_atom_mask: torch.Tensor,  # (N, 37(14))
        residue_index: torch.Tensor,  # (N)
        max_angstrom_tolerance=1.5,
        eps=1e-6,
) -> torch.Tensor:
    """Counts residues whose Ca is a large distance from its neighbour.

    Measures the fraction of CA-CA pairs between consecutive amino acids that are
    more than 'max_angstrom_tolerance' apart.

    Args:
      pred_atom_positions: Atom positions in atom37/14 representation
      pred_atom_mask: Atom mask in atom37/14 representation
      residue_index: Residue index for given amino acid, this is assumed to be
        monotonically increasing.
      max_angstrom_tolerance: Maximum distance allowed to not count as violation.
      eps: epsilon for numerical stability.
    Returns:
      Fraction of consecutive CA-CA pairs with violation.

    TODO: check if this works with only backbone atoms, from the looks of it, it does
    """
    this_ca_pos = pred_atom_positions[..., :-1, 1, :]
    this_ca_mask = pred_atom_mask[..., :-1, 1]
    next_ca_pos = pred_atom_positions[..., 1:, 1, :]
    next_ca_mask = pred_atom_mask[..., 1:, 1]
    has_no_gap_mask = (residue_index[..., 1:] - residue_index[..., :-1]) == 1.0
    ca_ca_distance = torch.sqrt(
        eps + torch.sum((this_ca_pos - next_ca_pos) ** 2, dim=-1)
    )
    violations = (
                         ca_ca_distance - residue_constants.ca_ca
                 ) > max_angstrom_tolerance
    mask = this_ca_mask * next_ca_mask * has_no_gap_mask
    mean = masked_mean(mask, violations, -1)
    return mean
