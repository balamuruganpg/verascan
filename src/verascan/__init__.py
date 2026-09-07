"""
Verascan — detect data contamination between training and evaluation sets.

Usage::

    import verascan

    report = verascan.check(train="train.jsonl", eval="eval.jsonl")
    report.summary()
"""

import verascan._env  # noqa: F401
from verascan._version import __version__
from verascan.audit import AuditReport, audit
from verascan.core import check
from verascan.report import ContaminationReport, MatchRecord
from verascan.split import split

__all__ = [
    "AuditReport",
    "ContaminationReport",
    "MatchRecord",
    "__version__",
    "audit",
    "check",
    "split",
]
