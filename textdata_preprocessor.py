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
"""
IU Xray medical report preprocessor for the text that does extraction
of findings and impression part of the text. Tokenize the sentence, lowercase
all the characters and remove the word contains non-alphabetic characters
"""
from typing import Iterator, Any
import os.path as osp
import argparse
import xml.etree.ElementTree as ET
from collections import Counter

from forte.data.data_pack import DataPack
from forte.data.readers.base_reader import PackReader
from forte.pipeline import Pipeline
from forte.data.multi_pack import MultiPack
from forte.processors.writers import PackNameJsonPackWriter
from forte.data.caster import MultiPackBoxer
from forte.data.selector import NameMatchSelector
from forte.data.data_utils_io import dataset_path_iterator

from iu_xray.onto import Impression, Findings, FilePath


class IUXrayReportReader(PackReader):
    r"""Customized reader for IU Xray report that read xml iteratively
    from the directory. Extract Findings and Impression from the reports.
    Remove all non-alpha tokens.
    """
    def _collect(self, text_directory) -> Iterator[Any]:
        r"""Should be called with param ``text_directory`` which is a path to a
        folder containing xml files.

        Args:
            text_directory: text directory containing the files.

        Returns: Iterator over paths to .xml files
        """
        return dataset_path_iterator(text_directory, '.xml')

    def _cache_key_function(self, xml_file: str) -> str:
        return osp.basename(xml_file)

    def _parse_pack(self, file_path: str) -> Iterator[DataPack]:

        tree = ET.parse(file_path)
        root = tree.getroot()

        counter = self.resources.get('counter')
        extracted = {'FINDINGS': None, 'IMPRESSION': None}
        to_find = 'MedlineCitation/Article/Abstract'
        for abs_text in root.find(to_find): # type: ignore
            label = abs_text.attrib['Label']
            if label in ['FINDINGS', 'IMPRESSION']:
                if abs_text.text is not None:
                    text = abs_text.text
                    text = text.replace(',', '')
                    text = [w for w in text.split()
                            if w.isalpha() or w[:-1].isalpha()
                            or w == '.']

                    text = ' '.join(text).lower()
                    extracted[label] = text

                    # Add words to the word counter
                    counter.update(
                        text.replace(',', '').replace('.', '').split(' '))

        for node in list(root):
            # One image report may consist of more that one
            # parent image (frontal, lateral)
            if node.tag == 'parentImage':
                try:
                    file_name = node.find('./panel/url').text # type: ignore
                except AttributeError:
                    msg = 'Cannot find the corresponding parent image'
                    raise ValueError(msg)

                pack = DataPack()
                # Findings
                findings = Findings(pack)
                findings.content = extracted['FINDINGS']
                # Impression
                impression = Impression(pack)
                impression.content = extracted['IMPRESSION']
                # FilePath
                filepath = FilePath(pack)

                pack_name_string = file_name.replace('.jpg', '')
                pack_name_list = pack_name_string.split('/')
                pack.pack_name = '_'.join(pack_name_list[-1:])
                filepath.img_study_path = '/'.join(pack_name_list[-1:])

                yield pack

def build_pipeline(result_dir: str,):
    r"""Build the pipeline to parse IU Xray report with tokenizer, lowercase and
    non-alpha removal to generate forte json file with the same name with
    preprocessed content and information of impression, findings and path to the
    parent image.
    Args:
        result_dir: the directory to save the forte json files.
    Return:
        pipeline: built pipeline to process the xml files
    """

    pipeline = Pipeline[MultiPack]()
    pipeline.resource.update(counter=Counter())
    pipeline.set_reader(IUXrayReportReader())
    pipeline.add(MultiPackBoxer())
    pipeline.add(PackNameJsonPackWriter(),
                 {'indent': 2, 'output_dir': result_dir, 'overwrite': True},
                 NameMatchSelector(select_name='default'))
    pipeline.initialize()

    return pipeline

def bulid_vocab(counter, save_vocab_dir: str):
    r"""Build the vocabulary using the top 1000 frequent words as per the paper
    Args:
        save_vocab_dir: the directory to to save the vocabulary (.txt).
    """
    words = [word for word, cnt in counter.most_common() if word != '']
    words = words[:1000]

    with open(save_vocab_dir, 'w') as f:
        for word in words:
            f.write(word)
            f.write("\n")
        f.close()


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--data-dir", type=str,
                        default="/data/jiachen.li/iu_xray/ecgen-radiology/",
                        help="Data directory to read the xml files from")
    PARSER.add_argument("--result-dir", type=str,
                        default="/home/jiachen.li/text_root/",
                        help="Data directory to save the forte json files to")
    PARSER.add_argument("--save-vocab-dir", type=str,
                        default="./texar_vocab.txt",
                        help="Directory to save the vocabulary (.txt)")

    ARGS = PARSER.parse_args()

    pl = build_pipeline(ARGS.result_dir)
    for idx, _ in enumerate(pl.process_dataset(ARGS.data_dir)):
        if (idx + 1) % 100 == 0:
            print("Processed " + str(idx + 1) + "packs")

    word_counter = pl.resource.get('counter')
    bulid_vocab(word_counter, ARGS.save_vocab_dir)
