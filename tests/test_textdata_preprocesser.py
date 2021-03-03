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
from textdata_preprocessor import build_pipeline
from forte.data.data_pack import DataPack


class TestBuildPipeline(unittest.TestCase):
    r"""
    Test iu xray processor for non alpha token removal in content
    """
    def setUp(self):
        self.result_dir = 'tests/test_xml/generated_xml'
        self.iu_xray_pl = build_pipeline(self.result_dir)

        self.ground_truth_findings = ''.join([
            'the lungs and pleural spaces show no acute abnormality. '
            'lungs are mildly hyperexpanded. heart size and pulmonary '
            'vascularity within normal limits.'
        ])
        self.ground_truth_impression = 'no acute pulmonary abnormality.'

    def test_pipeline(self):
        for _ in self.iu_xray_pl.process_dataset('tests/test_xml'):
            pass

        for i, filename in enumerate(os.listdir(self.result_dir)):
            self.assertIn(filename, ['CXR333_IM-1594-1001.json', 'CXR333_IM-1594-2001.json'])
            with open(osp.join(self.result_dir, filename), 'r') as f:
                items = list(DataPack.deserialize(f.read()))
                key = filename.replace('.json', '')
                self.assertEqual(items[0].img_study_path, key)
                self.assertEqual(items[1].content, self.ground_truth_findings)
                self.assertEqual(items[2].content, self.ground_truth_impression)


if __name__ == "__main__":
    unittest.main()
