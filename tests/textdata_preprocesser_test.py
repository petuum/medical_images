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
"""This module tests iu xray processor."""

import os
import os.path as osp
import unittest
from ft.onto.base_ontology import Sentence
from textdata_preprocessor import FindingsExtractor, ImpressionExtractor,\
    ParentImageExtractor, NonAlphaTokenRemover, build_pipeline
from iu_xray.onto import Impression, Findings, ParentImage
from forte.pipeline import Pipeline
from forte.data.data_pack import DataPack
from forte.data.readers import StringReader
from forte.data.caster import MultiPackBoxer
from forte.data.multi_pack import MultiPack
from forte.processors.nltk_processors import NLTKWordTokenizer,\
    NLTKSentenceSegmenter


class TestParentImageExtractor(unittest.TestCase):
    r"""
    Test iu xray processor for report impression section extractor
    """
    def setUp(self):
        self.iu_xray_pl = Pipeline[DataPack]()
        self.iu_xray_pl.set_reader(StringReader())
        self.iu_xray_pl.add(ParentImageExtractor())
        self.iu_xray_pl.initialize()

    def test_extractor(self):
        sentences = ["PARENTIMAGE /home/jiachen.li/iu_xray_images/CXR5_IM-2117-1003002.png "
                     "FINDINGS The cardiomediastinal silhouette and pulmonary vasculature "
                     "are within normal limits. There is no pneumothorax or pleural effusion. "
                     "There are no focal areas of consolidation. Cholecystectomy clips are present. "
                     "Small osteophytes. There is biapical pleural thickening unchanged from prior. "
                     "Mildly hyperexpanded lungs. IMPRESSION No acute cardiopulmonary abnormality."]
        document = ''.join(sentences)
        parent_image_sentences = ["/home/jiachen.li/iu_xray_images/CXR5_IM-2117-1003002.png"]
        parent_image_text = ''.join(parent_image_sentences)
        pack = self.iu_xray_pl.process(document)
        for idx, parent_image in enumerate(pack.get(ParentImage)):
            self.assertEqual(parent_image.text, parent_image_text)


class TestFindingsExtractor(unittest.TestCase):
    r"""
    Test iu xray processor for report findings section extractor
    """
    def setUp(self):
        self.iu_xray_pl = Pipeline[DataPack]()
        self.iu_xray_pl.set_reader(StringReader())
        self.iu_xray_pl.add(FindingsExtractor())
        self.iu_xray_pl.initialize()

    def test_extractor(self):
        sentences = ["PARENTIMAGE /home/jiachen.li/iu_xray_images/CXR5_IM-2117-1003002.png "
                     "FINDINGS The cardiomediastinal silhouette and pulmonary vasculature "
                     "are within normal limits. There is no pneumothorax or pleural effusion. "
                     "There are no focal areas of consolidation. Cholecystectomy clips are present. "
                     "Small osteophytes. There is biapical pleural thickening unchanged from prior. "
                     "Mildly hyperexpanded lungs. IMPRESSION No acute cardiopulmonary abnormality."]
        document = ''.join(sentences)
        findings_sentences = ["The cardiomediastinal silhouette and pulmonary vasculature "
                              "are within normal limits. There is no pneumothorax or pleural effusion. ",
                              "There are no focal areas of consolidation. Cholecystectomy clips are present. "
                              "Small osteophytes. There is biapical pleural thickening unchanged from prior. "
                              "Mildly hyperexpanded lungs."]
        findings_text = ''.join(findings_sentences)
        pack = self.iu_xray_pl.process(document)
        for idx, findings in enumerate(pack.get(Findings)):
            self.assertEqual(findings.text, findings_text)


class TestImpressionExtractor(unittest.TestCase):
    r"""
    Test iu xray processor for report impression section extractor
    """
    def setUp(self):
        self.iu_xray_pl = Pipeline[DataPack]()
        self.iu_xray_pl.set_reader(StringReader())
        self.iu_xray_pl.add(ImpressionExtractor())
        self.iu_xray_pl.initialize()

    def test_extractor(self):
        sentences = ["PARENTIMAGE /home/jiachen.li/iu_xray_images/CXR5_IM-2117-1003002.png "
                     "FINDINGS The cardiomediastinal silhouette and pulmonary vasculature "
                     "are within normal limits. There is no pneumothorax or pleural effusion. "
                     "There are no focal areas of consolidation. Cholecystectomy clips are present. "
                     "Small osteophytes. There is biapical pleural thickening unchanged from prior. "
                     "Mildly hyperexpanded lungs. IMPRESSION No acute cardiopulmonary abnormality."]
        document = ''.join(sentences)
        impression_sentences = ["No acute cardiopulmonary abnormality."]
        impression_text = ''.join(impression_sentences)
        pack = self.iu_xray_pl.process(document)
        for idx, impression in enumerate(pack.get(Impression)):
            self.assertEqual(impression.text, impression_text)


class TestNonAlphaTokenRemover(unittest.TestCase):
    r"""
    Test iu xray processor for non alpha token removal in content
    """
    def setUp(self):
        self.iu_xray_pl = Pipeline[MultiPack]()
        self.iu_xray_pl.set_reader(StringReader())
        self.iu_xray_pl.add(NLTKSentenceSegmenter())
        self.iu_xray_pl.add(NLTKWordTokenizer())
        self.iu_xray_pl.add(MultiPackBoxer())
        self.iu_xray_pl.add(NonAlphaTokenRemover())
        self.iu_xray_pl.initialize()

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
        pack = self.iu_xray_pl.process(document)
        for idx, sentences in enumerate(pack.get(Sentence)):
            self.assertEqual(sentences.text, rm_text)


class TestBuildPipeline(unittest.TestCase):
    r"""
    Test iu xray processor for non alpha token removal in content
    """
    def setUp(self):
        self.result_dir = 'tests/test_xml/generated_xml'
        self.img_root = '/home/jiachen.li/iu_xray_images/'
        self.iu_xray_pl = build_pipeline(
            self.result_dir,
            self.img_root)

        self.ground_truth_findings = ''.join([
            'the lungs and pleural spaces show no acute abnormality. '
            'lungs are mildly hyperexpanded. heart size and pulmonary '
            'vascularity within normal limits.'
        ])
        self.ground_truth_impression = 'no acute pulmonary abnormality.'

    def test_pipeline(self):
        # Generate the .json files
        for _ in self.iu_xray_pl.process_dataset('tests/test_xml'):
            pass

        for i, filename in enumerate(os.listdir(self.result_dir)):
            self.assertIn(filename, ['CXR333_IM-1594-1001.json', 'CXR333_IM-1594-2001.json'])
            with open(osp.join(self.result_dir, filename), 'r') as f:
                items = list(DataPack.deserialize(f.read()))
                key = filename.replace('.json', '')
                self.assertEqual(items[3].img_study_path, key)
                self.assertEqual(items[1].text, self.ground_truth_findings)
                self.assertEqual(items[2].text, self.ground_truth_impression)
                self.assertEqual(
                    items[0].text,
                    osp.join(self.img_root, key + '.png'))
                f.close()


if __name__ == "__main__":
    unittest.main()
