# ***automatically_generated***
# ***source json:../../../../../../medical_images_iu_xray/iu_xray_ontology.json***
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
    "ParentImage",
    "FilePath",
]


@dataclass
class Findings(Generics):
    """
    A span based annotation class Findings, used to refer to findings part of the report
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
    A span based annotation class Impression, used to refer to impression part of the report
    Attributes:
        content (Optional[str])
    """

    content: Optional[str]

    def __init__(self, pack: DataPack):
        super().__init__(pack)
        self.content: Optional[str] = None


@dataclass
class ParentImage(Generics):
    """
    A class ParentImage, used to refer to filepath of the parent images
    Attributes:
        parent_img_path (Optional[str])
    """

    parent_img_path: Optional[str]

    def __init__(self, pack: DataPack):
        super().__init__(pack)
        self.parent_img_path: Optional[str] = None


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
