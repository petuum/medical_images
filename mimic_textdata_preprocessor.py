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
mimic-cxr 2.0.0 medical report preprocessor for the text that does extraction
of findings and impression part of the text. Tokenize the sentence, lowercase
all the characters and remove the word contains non-alphabetic characters
"""
import argparse
from ft.onto.base_ontology import Token
from mimic.onto.mimic_ontology import Impression, Findings, FilePath
from forte.data.data_pack import DataPack
from forte.processors.base import PackProcessor
from forte.data.readers import PlainTextReader
from forte.data.span import Span
from forte.pipeline import Pipeline
from forte.processors.nltk_processors import NLTKWordTokenizer
from forte.data.multi_pack import MultiPack
from forte.processors.base import MultiPackProcessor
from forte.processors.writers import PackNameJsonPackWriter
from forte.data.caster import MultiPackBoxer
from forte.data.selector import NameMatchSelector


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
        findings = Findings(input_pack, begin, end)
        findings.has_content = (findings_ind != -1)


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
        impression = Impression(input_pack, begin, end)
        impression.has_content = (impression_ind != -1)


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
                 if word.isalpha() or word[:-1].isalpha() or word == '.']
        pack = input_pack.add_pack(self.out_pack_name)
        result = ' '.join(words)
        pack.set_text(text=result)
        filepath = FilePath(pack)
        if pack_name_string is not None:
            pack_name_string = pack_name_string.replace('.txt', '')
            pack_name_list = pack_name_string.split('/')
            pack.pack_name = '_'.join(pack_name_list[-3:])
            filepath.img_study_path = '/'.join(pack_name_list[-3:])
        else:
            pack.pack_name = 'packname'
            filepath.img_study_path = 'study_path'


class MimicReportReader(PlainTextReader):
    r"""Customized reader for mimic report that read text iteratively
    from the directory and replace non impression and non findings text with
    blank string.

    """

    @staticmethod
    def text_replace_operation(text: str):
        r"""Replace non impression and non findings text with blank string.
        Args:
            text: The original mimic report text to be cleaned.
        Returns: List[Tuple[Span, str]]: the replacement operations.
        """
        findings_ind = text.find("FINDINGS")
        impression_ind = text.find("IMPRESSION")
        end = len(text)
        replace_list = list()
        if findings_ind == -1 and impression_ind == -1:
            start = end
        elif findings_ind == -1 or impression_ind == -1:
            start = max(findings_ind, impression_ind)
        else:
            start = min(findings_ind, impression_ind)
        replace_list.append((Span(0, start), " "))

        return replace_list


def parse_mimic_reports(dataset_dir: str, result_dir: str):
    r"""Parse mimic report with tokenizer, lowercase and non-alpha removal to
    generate forte json file with the same name with preprocessed content and
    the span information of impression and findings.
    Args:
        dataset_dir: the directory that stores all the text files.
    """
    pipeline = Pipeline[MultiPack]()
    pipeline.set_reader(MimicReportReader())
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
    for idx, pack in enumerate(pipeline.process_dataset(dataset_dir)):
        if (idx + 1) % 50 == 0:
            print("Processed " + str(idx + 1) + "packs")


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--data-dir", type=str, default="data/",
                        help="Data directory to read the text files from")
    PARSER.add_argument("--result-dir", type=str, default="result/",
                        help="Data directory to save the forte json files to")
    ARGS = PARSER.parse_args()
    parse_mimic_reports(ARGS.data_dir, ARGS.result_dir)
