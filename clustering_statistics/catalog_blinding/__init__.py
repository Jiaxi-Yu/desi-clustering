"""Catalog-level blinding adapters for desi-clustering.

``desiblind`` owns the blinding physics and parameter banks. This subpackage owns
``desi-clustering`` adapters and LSS-like catalog workflow pieces: BAO/AP, RSD,
future fNL, random matching, n(z)/FKP updates, and the
``clustering-catalog-blinding`` CLI.
"""

from . import bao, rsd, fnl, lss_catalogs, diagnostics

__all__ = ['bao', 'rsd', 'fnl', 'lss_catalogs', 'diagnostics']
