"""Per-type elimination campaigns: screen types, then certify rivals below a bound.

``Campaign`` uses the level-2 SOS pipeline of :mod:`inscribed.sos` and stores certificates
per type in ``results/certificates``. ``ConvexCampaign`` adds convexity multipliers, works
from an exact committed ``TypeCatalogue`` and records a ``Ledger`` instead of certificates,
which are rebuilt and checked exactly on demand. Both resume where they stopped.
"""

from .campaign import Attempt, Campaign, Status
from .catalogue import ExactHull, TypeCatalogue, canonical_form, is_cycle_join
from .convex import ConvexCampaign, Ledger, LedgerEntry, below_two_pentagons
from .screening import Screening

__all__ = ["Attempt", "Campaign", "ConvexCampaign", "ExactHull", "Ledger", "LedgerEntry",
           "Screening", "Status", "TypeCatalogue", "below_two_pentagons", "canonical_form",
           "is_cycle_join"]
