from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


@dataclass
class ScenarioConfig:
    neutral_drift_shift_q: float = 0.0
    bullish_drift_shift_q: float = 0.0075
    bearish_drift_shift_q: float = -0.0075
    volatility_scale: float = 1.0


@dataclass
class FitConfig:
    min_obs_group: int = 40
    prior_strength: float = 60.0
    ratio_winsor_lower_q: float = 0.01
    ratio_winsor_upper_q: float = 0.99
    nav_gate: float = 1.0
    omega_fit_clip: tuple[float, float] = (-0.5, 0.5)
    size_bins: Sequence[float] = field(default_factory=lambda: [float("-inf"), 200_000_000.0, float("inf")])
    size_labels: Sequence[str] = field(default_factory=lambda: ["<200m", ">=200m"])


@dataclass
class CalibrationConfig:
    holdout_quarters: int = 8
    min_obs_for_adjustment: int = 250
    significance_alpha: float = 0.1
    shrink_n: float = 800.0
    apply_stability_attenuation: bool = True
    stability_step_q: int = 2
    stability_min_points: int = 8
    p_rep_mult_clip: tuple[float, float] = (0.8, 1.08)
    p_draw_mult_clip: tuple[float, float] = (0.85, 1.20)
    rep_ratio_scale_clip: tuple[float, float] = (0.7, 1.6)
    draw_ratio_scale_clip: tuple[float, float] = (0.7, 1.4)
    omega_bump_clip: tuple[float, float] = (-0.02, 0.06)
    post_invest_draw_prob_clip: tuple[float, float] = (0.05, 1.0)
    post_invest_draw_ratio_clip: tuple[float, float] = (0.05, 1.0)
    pre_end_repay_min_obs: int = 40
    pre_end_repay_prob_clip: tuple[float, float] = (0.6, 1.0)
    pre_end_repay_ratio_clip: tuple[float, float] = (0.6, 1.0)
    lifecycle_min_obs: int = 30
    lifecycle_p_mult_clip: tuple[float, float] = (0.6, 1.6)
    lifecycle_ratio_mult_clip: tuple[float, float] = (0.5, 2.0)
    lifecycle_draw_p_mult_clip: tuple[float, float] = (0.5, 1.5)
    lifecycle_draw_ratio_mult_clip: tuple[float, float] = (0.5, 1.5)
    lifecycle_omega_bump_clip: tuple[float, float] = (-0.08, 0.08)
    lifecycle_nav_floor_clip: tuple[float, float] = (0.0, 1.5)
    lifecycle_nav_floor_quantile: float = 0.5


@dataclass
class SimConfig:
    n_sims: int = 300
    seed: int = 1234
    horizon_q: int = 24
    draw_ratio_cap: float = 1.0
    omega_clip: tuple[float, float] = (-0.5, 0.5)
    # One-factor Gaussian copula dependency within each fund across all cashflow draws.
    copula_enabled: bool = True
    copula_rho: float | None = None
    post_invest_draws_enabled: bool = True
    pre_end_repay_enabled: bool = True
    post_end_runoff_enabled: bool = True
    post_end_no_draws: bool = True
    lifecycle_rep_strength: float = 0.5
    lifecycle_draw_strength: float = 0.5
    lifecycle_omega_strength: float = 0.35
    lifecycle_nav_floor_strength: float = 0.45
    lifecycle_nav_floor_strength_mid: float = 0.25
    lifecycle_nav_floor_min_start_age_q: int = 8


@dataclass
class PipelineConfig:
    cutoff_quarter: str = "2025Q3"
    run_tag: str = "2025Q3_v2"
    scenario: str = "neutral"
    fit: FitConfig = field(default_factory=FitConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    simulation: SimConfig = field(default_factory=SimConfig)
    scenarios: ScenarioConfig = field(default_factory=ScenarioConfig)

    @property
    def run_root(self) -> Path:
        return Path("runs_v2") / self.run_tag

    @property
    def calibration_dir(self) -> Path:
        return self.run_root / "calibration"

    @property
    def projection_dir(self) -> Path:
        return self.run_root / "projection"
