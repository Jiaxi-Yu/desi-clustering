"""Future catalog-level fNL blinding adapter.

The actual fNL blinding physics should live in ``desiblind``. This module is the
future ``desi-clustering`` saved-catalog adapter layer: catalog I/O, option
plumbing, and calls into a future desiblind fNL blinder.
"""


def not_implemented(*args, **kwargs):
    """Raise a clear error for the planned-but-unimplemented fNL path."""
    raise NotImplementedError(
        'Catalog-level fNL blinding is not implemented yet. Add the desiblind fNL blinder first, '
        'then wire it through clustering_statistics.catalog_blinding.fnl and the saved-catalog driver.'
    )
