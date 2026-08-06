"""Package entry points for the DJGP implementation.

The repository is being migrated from flat scripts to packages. Submodules in
this package currently wrap the legacy modules so old entrypoints keep working.
"""

__all__ = [
    "active_learning",
    "acquisition_metrics",
    "jumpgp_bridge",
    "minibatch",
    "sir",
    "variational",
]
