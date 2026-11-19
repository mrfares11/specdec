"""Config loading: OmegaConf YAML merge + dotlist overrides.

Convention used by every script:

    python scripts/<x>.py --config configs/base.yaml [configs/local.yaml ...] \
        --set K=12 tree.allocation=geometric paths.root=./runs

Files merge left-to-right; ``--set`` dotlist entries win over everything.
"""

from __future__ import annotations

import argparse
from typing import Sequence

from omegaconf import DictConfig, OmegaConf


def load_config(
    config_paths: Sequence[str],
    overrides: Sequence[str] = (),
) -> DictConfig:
    """Merge YAML files left-to-right, then apply dotlist overrides.

    Args:
        config_paths: YAML file paths, later files override earlier ones.
        overrides: OmegaConf dotlist entries like ``"K=12"``.

    Returns:
        The merged config (interpolations such as ``${K}`` stay live, so they
        reflect overrides applied here).
    """
    if not config_paths:
        raise ValueError("at least one config file is required")
    cfgs = [OmegaConf.load(str(p)) for p in config_paths]
    dot = OmegaConf.from_dotlist([str(o) for o in overrides])
    merged = OmegaConf.merge(*cfgs, dot)
    assert isinstance(merged, DictConfig)
    return merged


def parse_cli(
    description: str,
    default_configs: Sequence[str] = ("configs/base.yaml",),
) -> DictConfig:
    """Standard script CLI: ``--config <yaml...> --set key=value ...``."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument(
        "--config", nargs="+", default=list(default_configs), metavar="YAML",
        help="config files, merged left-to-right",
    )
    ap.add_argument(
        "--set", nargs="*", default=[], dest="overrides", metavar="KEY=VALUE",
        help="dotlist overrides (highest priority)",
    )
    args = ap.parse_args()
    return load_config(args.config, args.overrides)
