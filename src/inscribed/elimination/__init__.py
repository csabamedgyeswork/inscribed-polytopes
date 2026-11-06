"""Per-type elimination campaigns: screen types, then certify rivals below a bound.

Uses the level-2 SOS pipeline of :mod:`inscribed.sos`; results are stored per type
in the existing ``results/certificates`` tree, so a campaign resumes where it stopped
and can be split into shards across machines.
"""

from .campaign import Attempt, Campaign, Status
from .screening import Screening

__all__ = ["Attempt", "Campaign", "Screening", "Status"]
