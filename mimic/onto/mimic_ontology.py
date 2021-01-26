# ***automatically_generated***
# ***source json:../../../../../../home/jenny.zhang/medical_images/mimic_ontology.json***
# flake8: noqa
# mypy: ignore-errors
# pylint: skip-file
"""
Automatically generated ontology mimic_ontology. Do not change manually.
"""

from dataclasses import dataclass
from forte.data.data_pack import DataPack
from forte.data.ontology.top import Annotation
from typing import Optional

__all__ = [
    "Findings",
    "Impression",
    "FilePath",
]


@dataclass
class Findings(Annotation):
    """
    A span based annotation class Findings, used to refer to findings part of the report
    Attributes:
        has_content (Optional[bool])
    """

    has_content: Optional[bool]

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self.has_content: Optional[bool] = None


@dataclass
class Impression(Annotation):
    """
    A span based annotation class Impression, used to refer to impression part of the report
    Attributes:
        has_content (Optional[bool])
    """

    has_content: Optional[bool]

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self.has_content: Optional[bool] = None


@dataclass
class FilePath(Annotation):
    """
    A class used to refer to file path hierarchy of the report
    Attributes:
        path (Optional[str])
    """

    path: Optional[str]

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self.path: Optional[str] = None
