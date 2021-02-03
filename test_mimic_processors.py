# Copyright 2021 The Petuum Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# pylint: disable=attribute-defined-outside-init
"""This module tests mimic processor."""

import unittest
from ft.onto.base_ontology import Sentence
from textdata_reader import FindingsExtractor, ImpressionExtractor,\
    NonAlphaTokenRemover
from mimic.onto.mimic_ontology import Impression, Findings
from forte.pipeline import Pipeline
from forte.data.data_pack import DataPack
from forte.data.readers import StringReader
from forte.data.caster import MultiPackBoxer
from forte.data.multi_pack import MultiPack
from forte.processors.nltk_processors import NLTKWordTokenizer,\
    NLTKSentenceSegmenter


class TestFindingsExtractor(unittest.TestCase):
    r"""
    Test mimic processor for report findings section extractor
    """
    def setUp(self):
        self.mimic_pl = Pipeline[DataPack]()
        self.mimic_pl.set_reader(StringReader())
        self.mimic_pl.add(FindingsExtractor())
        self.mimic_pl.initialize()

    def test_extractor(self):
        sentences = ["FINDINGS There is no focal consolidation "
                     "pleural effusion or pneumothorax.",
                     "Bilateral nodular opacities that most likely "
                     "represent nipple shadows.",
                     "The cardiomediastinal silhouette is normal.",
                     "Clips project over the left lung "
                     "potentially within the breast.",
                     "The imaged upper abdomen is unremarkable.",
                     "Chronic deformity of the posterior left "
                     "sixth and seventh ribs are noted.",
                     "IMPRESSION No acute cardiopulmonary process."]
        document = ' '.join(sentences)
        findings_sentences = ["There is no focal consolidation "
                              "pleural effusion or pneumothorax.",
                              "Bilateral nodular opacities that most likely "
                              "represent nipple shadows.",
                              "The cardiomediastinal silhouette is normal.",
                              "Clips project over the left lung "
                              "potentially within the breast.",
                              "The imaged upper abdomen is unremarkable.",
                              "Chronic deformity of the posterior left "
                              "sixth and seventh ribs are noted."]
        findings_text = ' '.join(findings_sentences)
        pack = self.mimic_pl.process(document)
        for idx, findings in enumerate(pack.get(Findings)):
            self.assertEqual(findings.text, findings_text)


class TestImpressionExtractor(unittest.TestCase):
    r"""
    Test mimic processor for report impression section extractor
    """
    def setUp(self):
        self.mimic_pl = Pipeline[DataPack]()
        self.mimic_pl.set_reader(StringReader())
        self.mimic_pl.add(ImpressionExtractor())
        self.mimic_pl.initialize()

    def test_extractor(self):
        sentences = ["FINDINGS There is no focal consolidation "
                     "pleural effusion or pneumothorax.",
                     "Bilateral nodular opacities that most likely "
                     "represent nipple shadows.",
                     "The cardiomediastinal silhouette is normal.",
                     "Clips project over the left lung "
                     "potentially within the breast.",
                     "The imaged upper abdomen is unremarkable.",
                     "Chronic deformity of the posterior left "
                     "sixth and seventh ribs are noted.",
                     "IMPRESSION No acute cardiopulmonary process."]
        document = ' '.join(sentences)
        impression_sentences = ["No acute cardiopulmonary process."]
        impression_text = ' '.join(impression_sentences)
        pack = self.mimic_pl.process(document)
        for idx, impression in enumerate(pack.get(Impression)):
            self.assertEqual(impression.text, impression_text)


class TestNonAlphaTokenRemover(unittest.TestCase):
    r"""
    Test mimic processor for non alpha token removal in content
    """
    def setUp(self):
        self.mimic_pl = Pipeline[MultiPack]()
        self.mimic_pl.set_reader(StringReader())
        self.mimic_pl.add(NLTKSentenceSegmenter())
        self.mimic_pl.add(NLTKWordTokenizer())
        self.mimic_pl.add(MultiPackBoxer())
        self.mimic_pl.add(NonAlphaTokenRemover())
        self.mimic_pl.initialize()

    def test_cleaner(self):
        sentences = ["Free intraperitoneal air.  ___ was successfully "
                     "paged to discuss this finding on ___ at 3:10 p.m. "
                     "at the time of discovery."]
        document = ' '.join(sentences)
        nonalpha_token_rm_sentences = ["Free intraperitoneal air. was "
                                       "successfully paged to discuss "
                                       "this finding on at pm at the "
                                       "time of discovery."]

        rm_text = ' '.join(nonalpha_token_rm_sentences)
        pack = self.mimic_pl.process(document)
        for idx, sentences in enumerate(pack.get(Sentence)):
            self.assertEqual(sentences.text, rm_text)
