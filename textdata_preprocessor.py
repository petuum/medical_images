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

from ft.onto.base_ontology import Token, Document
from forte.data.data_pack import DataPack
from forte.processors.base import PackProcessor
from forte.data.readers.base_reader import PackReader
from forte.pipeline import Pipeline
from forte.processors.nltk_processors import NLTKWordTokenizer
from forte.data.multi_pack import MultiPack
from forte.processors.base import MultiPackProcessor
from forte.processors.writers import PackNameJsonPackWriter
from forte.data.caster import MultiPackBoxer
from forte.data.selector import NameMatchSelector
from forte.data.data_utils_io import dataset_path_iterator

from iu_xray.onto import Impression, Findings, FilePath


class FindingsExtractor(PackProcessor):
    r"""A processor to extract the Findings session in the medical report.
    """

    def _process(self, input_pack: DataPack):
        findings_ind = input_pack.text.find("FINDINGS")
        impression_ind = input_pack.text.find("IMPRESSION")
        if findings_ind == -1:
            begin = 0
            end = 0
        else:
            begin = findings_ind + len("FINDINGS") + 1
            if impression_ind != -1 and findings_ind < impression_ind:
                end = impression_ind - 1
            else:
                end = len(input_pack.text)

        if end > begin:
            findings = Findings(input_pack, begin, end)
            findings.has_content = True
        else:
            findings = Findings(input_pack, end, end)
            findings.has_content = False

        counter = self.resources.get('counter')
        if findings.has_content and counter is not None:
            # Update the vocabulary counter
            text = input_pack.text[begin: end].replace(',', '').replace('.', '')
            counter.update(text.split(' '))


class ImpressionExtractor(PackProcessor):
    r"""A processor to extract the Impression session in the medical report.
    """

    def _process(self, input_pack: DataPack):
        findings_ind = input_pack.text.find("FINDINGS")
        impression_ind = input_pack.text.find("IMPRESSION")
        if impression_ind == -1:
            begin = 0
            end = 0
        else:
            begin = impression_ind + len("IMPRESSION") + 1
            if findings_ind != -1 and impression_ind < findings_ind:
                end = findings_ind - 1
            else:
                end = len(input_pack.text)

        if end > begin:
            impression = Impression(input_pack, begin, end)
            impression.has_content = True
        else:
            impression = Impression(input_pack, end, end)
            impression.has_content = False

        counter = self.resources.get('counter')
        if impression.has_content and counter is not None:
            # Update the vocabulary counter
            text = input_pack.text[begin: end].replace(',', '').replace('.', '')
            counter.update(text.split(' '))


class NonAlphaTokenRemover(MultiPackProcessor):
    r"""A class of non alpha token remover that requires a nltk tokenizer in the
    upstream of the pipeline. The modified text would be added to a new pack.
    """

    def __init__(self):
        super().__init__()
        self.in_pack_name = 'default'
        self.out_pack_name = 'result'

    def _process(self, input_pack: MultiPack):
        pack_name_string = input_pack.get_pack(self.in_pack_name).pack_name

        token_entries = list(input_pack.get_pack(
            self.in_pack_name).get(entry_type=Token))
        token_texts = [token.text for token in token_entries]
        words = [word for word in token_texts
                 if word.isalpha() or word[:-1].isalpha()
                 or word == '.' or osp.isfile(word)]

        if words[-1] == '.':
            # Combine the last '.' with the final word
            words = words[:-1]
            words[-1] += '.'

        pack = input_pack.add_pack(self.out_pack_name)
        result = ' '.join(words)

        pack.set_text(text=result)
        filepath = FilePath(pack)

        if pack_name_string is not None:
            pack_name_string = pack_name_string.replace('.jpg', '')
            pack_name_list = pack_name_string.split('/')
            pack.pack_name = '_'.join(pack_name_list[-1:])
            filepath.img_study_path = '/'.join(pack_name_list[-1:])
        else:
            pack.pack_name = 'packname'
            filepath.img_study_path = 'study_path'


class IUXrayReportReader(PackReader):
    r"""Customized reader for IU Xray report that read xml iteratively
    from the directory.
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

        abs_text_list = []
        for abs_text in root.find('MedlineCitation/Article/Abstract'):
            if abs_text.attrib['Label'] in ['FINDINGS', 'IMPRESSION']:
                text = abs_text.text if abs_text.text else ' '
                content = abs_text.attrib['Label'] + ' ' + text.lower()
                abs_text_list.append(content)

        for node in list(root):
            # One image report may consist of more that one
            # parent image (frontal, lateral)
            if node.tag == 'parentImage':
                try:
                    file_name = node.find('./panel/url').text
                except AttributeError:
                    raise AttributeError
                text = ' '.join(abs_text_list)
                pack = DataPack()
                pack.set_text(text)

                Document(pack, 0, len(pack.text))
                pack.pack_name = file_name

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
    pipeline.add(NLTKWordTokenizer())
    pipeline.add(MultiPackBoxer())
    pipeline.add(NonAlphaTokenRemover())
    pipeline.add(component=FindingsExtractor(),
                 selector=NameMatchSelector(select_name='result'))
    pipeline.add(component=ImpressionExtractor(),
                 selector=NameMatchSelector(select_name='result'))
    pipeline.add(PackNameJsonPackWriter(),
                 {'indent': 2, 'output_dir': result_dir, 'overwrite': True},
                 NameMatchSelector(select_name='result'))
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
