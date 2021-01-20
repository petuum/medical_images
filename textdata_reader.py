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
from forte.data.readers import PlainTextReader
from forte.data.span import Span
from forte.pipeline import Pipeline
from forte.processors.nltk_processors import NLTKWordTokenizer, \
    NLTKSentenceSegmenter
from ft.onto.base_ontology import Token, Sentence


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
    """
    :param dataset_dir: the directory that stores all the text files
    :param save_csv: the csv file to save the result
    :return:
    """
    pipeline = Pipeline[DataPack]()
    pipeline.set_reader(MimicReportReader())
    pipeline.add(NLTKSentenceSegmenter())
    pipeline.add(NLTKWordTokenizer())
    pipeline.initialize()
    dataframe = pd.DataFrame(columns=["path", "findings", "impression"])

    for pack in pipeline.process_dataset(dataset_dir):
        print(colored("Document", 'red'), pack.pack_name)
        findings_sentence = ""
        impression_sentence = ""
        findings_mode = False
        impression_mode = False
        for sentence in pack.get(Sentence):
            tokens = [token.text.lower() for token in
                      pack.get(Token, sentence)]
            if tokens[0] == 'findings':
                tokens = tokens[1:]
                findings_mode = True
                impression_mode = False
            elif tokens[0] == 'impression':
                tokens = tokens[1:]
                impression_mode = True
                findings_mode = False

            words = [word for word in tokens
                     if word.isalpha() or word[:-1].isalpha()]

            result = ' '.join(words)
            if findings_mode:
                findings_sentence += result
                findings_sentence += " "
            if impression_mode:
                impression_sentence += result
                impression_sentence += " "

        curr_path = pack.pack_name.replace(dataset_dir, "")
        dataframe = dataframe.append({"path": curr_path, "impression": impression_sentence,
                                      "findings": findings_sentence}, ignore_index=True)

    dataframe.to_csv(save_csv)


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--data-dir", type=str, default="data/",
                        help="Data directory to read the text files from")
    PARSER.add_argument("--save-csv", type=str, default="mimic_text_full.csv",
                        help="csv file to save the result")
    ARGS = PARSER.parse_args()
    parse_mimic_reports(ARGS.data_dir, ARGS.save_csv)
