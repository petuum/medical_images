# ***automatically_generated***
# ***source json:../../../../../../petuum-med/medical_images/iu_xray_ontology.json***
# flake8: noqa
# mypy: ignore-errors
# pylint: skip-file
"""
Automatically generated ontology iu_xray_ontology. Do not change manually.
"""

from dataclasses import dataclass
from forte.data.data_pack import DataPack
from forte.data.ontology.top import Generics
from typing import Optional

__all__ = [
    "Findings",
    "Impression",
    "FilePath",
]


@dataclass
class Findings(Generics):
    """
    A Generics class Findings, used to refer to findings part of the report
    Attributes:
        content (Optional[str])
    """

    content: Optional[str]

    def __init__(self, pack: DataPack):
        super().__init__(pack)
        self.content: Optional[str] = None


@dataclass
class Impression(Generics):
    """
    A Generics class Impression, used to refer to impression part of the report
    Attributes:
        content (Optional[str])
    """

    content: Optional[str]

    def __init__(self, pack: DataPack):
        super().__init__(pack)
        self.content: Optional[str] = None


@dataclass
class FilePath(Generics):
    """
    A class FilePath, used to refer to filepath of the report
    Attributes:
        img_study_path (Optional[str])
    """

    img_study_path: Optional[str]

    def __init__(self, pack: DataPack):
        super().__init__(pack)
        self.img_study_path: Optional[str] = None
