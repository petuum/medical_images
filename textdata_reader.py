# Copyright 2019 The Forte Authors. All Rights Reserved.
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
"""
mimic-cxr 2.0.0 medical report preprocessor for the text that does extraction of findings
and impression part of the text. Tokenize the sentence, lowercase all the characters and
remove the word contains non-alphabetic characters
"""
import argparse
import pandas as pd
from termcolor import colored
from forte.data.data_pack import DataPack
from forte.processors.base import PackProcessor
from forte.data.readers import PlainTextReader
from forte.data.span import Span
from forte.pipeline import Pipeline
from forte.processors.nltk_processors import NLTKWordTokenizer, \
    NLTKSentenceSegmenter
from forte.processors.lowercaser_processor import LowerCaserProcessor
from mimic.onto.mimic_ontology import Impression, Findings, FilePath


class FindingsExtractor(PackProcessor):
    r"""A wrapper of fingings extractor.
    """

    def __init__(self):
        super().__init__()

    def _process(self, input_pack: DataPack):
        findings_ind = input_pack.text.find("FINDINGS")
        impression_ind = input_pack.text.find("IMPRESSION")
        begin = 0 if findings_ind == -1 else findings_ind
        end = 0 if impression_ind == -1 else impression_ind - 1
        findings = Findings(input_pack, begin, end)
        findings.has_content = (begin == -1)

class ImpressionExtractor(PackProcessor):
    r"""A wrapper of impression extractor.
    """

    def __init__(self):
        super().__init__()

    def _process(self, input_pack: DataPack):
        impression_ind = input_pack.text.find("IMPRESSION")
        begin = 0 if impression_ind == -1 else impression_ind
        end = len(input_pack.text)
        impression = Impression(input_pack, begin, end)
        impression.has_content = (begin == -1)

class FilePathGetter(PackProcessor):
    r"""A wrapper to get the file path hierarchy of the current file
    """
    def __init__(self):
        super().__init__()

    def _process(self, input_pack: DataPack):
        file_path = FilePath(input_pack, 0, len(input_pack.text))
        file_path.path = input_pack.pack_name

# class NonAlphaTokenRemover(PackProcessor):
#     r"""A wrapper of NLTK word tokenizer.
#     """
#
#     def __init__(self):
#         super().__init__()
#
#     def _process(self, input_pack: DataPack):


class MimicReportReader(PlainTextReader):
    r"""Customized reader for mimic report that read text iteratively from the dir and
    Replace non impression and non findings text with blank string.

    """
    @staticmethod
    def text_replace_operation(text: str):
        r"""Replace non impression and non findings text with blank string.
        Args:
            text: The original mimic report text to be cleaned.
        Returns: List[Tuple[Span, str]]: the replacement operations
        """
        findings_ind = text.find("FINDINGS")
        impression_ind = text.find("IMPRESSION")
        end = len(text)
        if findings_ind == -1 and impression_ind == -1:
            start = end
        elif findings_ind == -1 or impression_ind == -1:
            start = max(findings_ind, impression_ind)
        else:
            start = min(findings_ind, impression_ind)

        return [(Span(0, start), " ")]


def parse_mimic_reports(dataset_dir: str, save_csv: str):
    r"""Parse mimic report with tokenizer, lowercase and non-alpha removal to
    generate csv with preprocessed impression and findings content
    Args:
        dataset_dir: the directory that stores all the text files
        save_csv: the csv file to save the result
    """
    pipeline = Pipeline[DataPack]()
    pipeline.set_reader(MimicReportReader())
    pipeline.add(FindingsExtractor())
    pipeline.add(ImpressionExtractor())
    pipeline.add(NLTKWordTokenizer())
    pipeline.add(FilePathGetter())
    pipeline.add(LowerCaserProcessor())
    pipeline.initialize()

    for pack in pipeline.process_dataset(dataset_dir):
        print(colored("Document", 'red'), pack.pack_name)
        for findings in pack.get_data(Findings):
            print(findings["context"])
        for impression in pack.get_data(Impression):
            print(impression["context"])

        # tokens = [token.text.lower() for token in
        #           pack.get(Token, sentence)]
        #
        # words = [word for word in tokens
        #          if word.isalpha() or word[:-1].isalpha()]
        #
        # result = ' '.join(words)

        curr_path = pack.pack_name.replace(dataset_dir, "")




if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--data-dir", type=str, default="data/",
                        help="Data directory to read the text files from")
    PARSER.add_argument("--save-csv", type=str, default="mimic_text_full.csv",
                        help="csv file to save the result")
    ARGS = PARSER.parse_args()
    parse_mimic_reports(ARGS.data_dir, ARGS.save_csv)
