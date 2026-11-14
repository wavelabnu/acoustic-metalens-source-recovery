"""
direct — the direct/forward problem: ~8 sources behind the lens radiate and are
recorded at a far line of microphones; recover the source and its resolution
limits from the mic data.

Forward model in `physics`, inverse/analysis in `recover`, and in `prior` the
price of prior information: what a decoder carrying a class prior renders past
the channel budget, and what that assumption is worth in bits. The notebooks
import from here instead of re-implementing the engine.

    from direct import Config, Lens, CASES, transfer_matrix, stacked_matrix
    from direct import analyze_T, mode_recovery_sweep, source_modes
"""
from .physics import (
    CDTYPE, NOISE_ABS, CASES, STYLES,
    Config, Lens, line_lens, green, gpref,
    transfer_matrix, stacked_matrix, solve_amplitudes,
    field_on_grid, mic_data, lens_collective_modes,
)
from .recover import (
    source_modes, truncated_pinv, recover_image,
    mode_recovery_sweep, analyze_T, mode_capture, pattern_budget_sweep,
)
from .prior import (
    real_stack, real_noise, support_dictionary, prior_bits, channel_bits,
    sparse_map, unconstrained_mmse, two_source_pattern, class_prior, prior_bits_gaussian, gaussian_mmse, recoverable_projector,
    apparent_k, prior_ledger, gaussian_information, gaussian_posterior_covariance,
    gaussian_mode_information, exact_supports, draw_exact_sparse,
    exact_sparse_posterior, exact_sparse_information, high_mode_fraction,
    support_hallucination_metrics, first_reliable_budget,
    channel_recovery_error, )

__all__ = [
    "CDTYPE", "NOISE_ABS", "CASES", "STYLES",
    "Config", "Lens", "line_lens", "green", "gpref",
    "transfer_matrix", "stacked_matrix", "solve_amplitudes",
    "field_on_grid", "mic_data", "lens_collective_modes",
    "source_modes", "truncated_pinv", "recover_image",
    "mode_recovery_sweep", "analyze_T", "mode_capture", "pattern_budget_sweep",
    "real_stack", "real_noise", "support_dictionary", "prior_bits",
    "channel_bits", "sparse_map", "unconstrained_mmse", "two_source_pattern",
    "class_prior", "prior_bits_gaussian", "gaussian_mmse",
    "recoverable_projector", "apparent_k", "prior_ledger",
    "gaussian_information", "gaussian_posterior_covariance",
    "gaussian_mode_information", "exact_supports", "draw_exact_sparse",
    "exact_sparse_posterior", "exact_sparse_information",
    "high_mode_fraction", "support_hallucination_metrics",
    "first_reliable_budget",
    "channel_recovery_error",
]
