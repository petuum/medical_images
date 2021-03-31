from typing import Dict, Any
import torch
import torch.nn as nn

from texar.torch import ModuleBase, HParams
from texar.torch.losses import sequence_sparse_softmax_cross_entropy
from texar.torch.evals import corpus_bleu
from texar.torch.utils import strip_special_tokens
from texar.torch.data import Vocab

from models.cv_model import MLC, SimpleFusionEncoder
from models.nlp_model import LstmSentence, LstmWord, SemanticTagGenerator


class MedicalReportGenerator(ModuleBase):
    r"""Medical report generator that generates medical reports
    from the input images. There are mainly six modules:
    a CNN feature extractor, a multi-label classifier (MLC),
    a SemanticTagGenerator, a CoAttentionmodule, a Sentence LSTM
    and a Word LSTM. It first extracts visual features from the input images.
    The MLC is then used to predict the probabilities of active tags. The
    SemanticTagGenerator selects the top K tags with the highest probabilites
    and generate corresponding semantic features. The CoAttention module then
    extracts context vector from both the visual and semantic features
    based on the previous state of the Sentence LSTM.

    By conditioning on the context vectors, a two 2 layered hierarchical
    LSTM are proposed to generate the actual paragraph. Specifically,
    the first layer is the Sentence LSTM that generate "topic" for
    each sentence by taking as input the context vector. The "topic" is
    passed to the Word LSTM and generate the content for each sentence.
    Moreover, the Sentence LSTM also learns to control the continuation
    of sentence generation.

    Args:
        hparams (Dict or HParams, optional): LstmWord hyperparameters.
            Missing hyperparameters will be set to default values.
            See :meth:`default_hparams` for the hyperparameter
            structure and default values.
                * sentence_lstm (Dict): Hyperparameters for the Sentence LSTM
                * tag_generator (Dict): Hyperparameters for the tag generator
                * word_lstm (Dict): Hyperparameters for the Word LSTM
                * lambda_stop (float): Weights for the stop control loss
                * lambda_word (float): Weights for the word prediction loss
                * lambda_word (float): Weights for the regularization loss
                * visual_weights (str): Directory to the weights of
                    pretrained CNN and MLC
                * train_visual (bool): Whether to train the feature extractor
                    and MLC
                * pathologies (List[str]): List of the tags
                * vocab_path (str): Directory to the txt file that contains all
                        the words for the vocabulary
    """
    def __init__(self, hparams):
        super().__init__()
        self._hparams = HParams(hparams, self.default_hparams())

        if self._hparams.visual_weights:
            self.visual_weights = torch.load(self._hparams.visual_weights).model
        else:
            self.visual_weights = None
        self.train_visual = self._hparams.train_visual

        self.params = None
        self.ce_criterion = nn.CrossEntropyLoss()
        self.pathologies = self._hparams.pathologies
        self.vocab = Vocab(self._hparams.vocab_path)

        self.extractor = self._init_extractor()
        self.mlc = self._init_mlc()
        self.tag_generator = self._init_tag_generator()
        self.sentence_lstm = self._init_sentence_lstm()
        self.word_lstm = self._init_word_lstm()

        self.lambda_stop = self._hparams.lambda_stop
        self.lambda_word = self._hparams.lambda_word
        self.lambda_attn = self._hparams.lambda_attn
        self.max_sent_num = self._hparams.max_sent_num

    def _init_model(self, model):
        r"""Add the parameters of the input model to self.params.
        Move the input model to GPU if it is applicable

        Args:
            model (Any): input model

        Returns:
            model (Any): model that moved to the desired device
        """
        if self.params:
            self.params += list(model.parameters())
        else:
            self.params = list(model.parameters())

        if torch.cuda.is_available():
            model = model.cuda()
        return model

    def _init_extractor(self):
        r"""Initialize the visual feature extractor. Load the weights
        from the checkpoint if it is available.

        NOTE: The checkpoint is resulted from separately training the
        MLC on the multi-label classification task.
        """
        model = SimpleFusionEncoder()
        if self.visual_weights is not None:
            m_state_dict = {}
            len_extractor = len('extractor.')
            for k in self.visual_weights.keys():
                if 'extractor.' in k:
                    new_k = k[len_extractor:]
                    m_state_dict[new_k] = self.visual_weights[k]

            model.load_state_dict(m_state_dict)

        if not self.train_visual:
            for p in model.parameters():
                p.requires_grad = False

        if torch.cuda.is_available():
            model = model.cuda()

        return model

    def _init_mlc(self):
        r"""Initialize the MLC. Load the weights from the checkpoint
        if it is available.

        NOTE: The checkpoint is resulted from separately training the
        MLC on the multi-label classification task.
        """
        model = MLC(self._hparams.mlc)

        if self.visual_weights is not None:
            m_state_dict = {}
            len_mlc = len('mlc.')

            for k in self.visual_weights.keys():
                if 'mlc.' in k:
                    new_k = k[len_mlc:]
                    m_state_dict[new_k] = self.visual_weights[k]

            model.load_state_dict(m_state_dict)

        if not self.train_visual:
            for p in model.parameters():
                p.requires_grad = False

        if torch.cuda.is_available():
            model = model.cuda()

        return model

    def _init_tag_generator(self):
        r"""Initialize the SemanticTagGenerator"""
        model = SemanticTagGenerator(self._hparams.tag_generator)
        model = self._init_model(model)

        return model

    def _init_sentence_lstm(self):
        r"""Initialize the Sentence LSTM"""
        model = LstmSentence(self._hparams.sentence_lstm)
        model = self._init_model(model)

        return model

    def _init_word_lstm(self):
        r"""Initialize the Word LSTM"""
        self._hparams.word_lstm.add_hparam("vocab_size", self.vocab.size)
        self._hparams.word_lstm.add_hparam("BOS", self.vocab.bos_token_id)
        self._hparams.word_lstm.add_hparam("EOS", self.vocab.eos_token_id)

        model = LstmWord(self._hparams.word_lstm)
        model = self._init_model(model)

        return model

    def attn_loss(self, visual_aligns, semantic_aligns):
        r"""Attention loss function. Weigh it in the
        main loop as per lambda

        Args:
            visual_aligns (torch.Tensor): Dimension
                [batch_size, num_visual_features, num_sentence_lstm_step]
            semantic_aligns (torch.Tensor): Dimension
                [batch_size, num_semantic_features, num_sentence_lstm_step]

        Returns:
            loss (torch.float): Calculated attention loss
        """

        visual_attn_loss = torch.sum(
            ((1 - torch.sum(visual_aligns, -1)) ** 2), -1)

        semantic_attn_loss = torch.sum(
            ((1 - torch.sum(semantic_aligns, -1)) ** 2), -1)

        return torch.mean(visual_attn_loss + semantic_attn_loss)

    def forward(self, batch):
        r"""Train the whole pipeline. Please refer to above for
        details. We used a Teacher Forcing manner to train the Word LSTM.

        NOTE: Different from the other seq2seq project, we use the topic
        and the special BOS token as the first and second input. We thus
        do not count the first word genarated by the Word LSTM. We need to
        set sentence_len = teacher_word_sequence_len + 1 since we do not
        count the first word. Moreover, we do not include the BOS token in
        the teacher words.

        We need to create the input as
        [Topic; Embedding of BOS; Embedding of Teacher's words]
        for the Word LSTM (BasicRNNDecoder). Therefore, we embed the teacher's
        words before passing it to the Word LSTM, and we set the token_embedder
        of it as Identity function.

        Args:
            batch (tx.torch.data.Batch[str, Union[torch.Tensor, int]]):
                * batch_size: batch size
                * label: Dimension [batch size, num_tags]
                * img_tensor: Dimension [batch size, channels, height, width]
                * token_tensor: Dimension
                    [batch size, max_sentence_num + 1, max_word_num]
                * stop_prob: Dimension [batch size, max_sentence_num + 1]

        Returns:
            loss (torch.float): sum of weighted losses,
            stop_loss (torch.float): stop control loss,
            word_loss (torch.float): word prediction loss,
            attention_loss (torch.float): regularization loss,
            topic_var (torch.float): averaged variance of the topics across
                different sample in the current batch
            bleu (1- 4) (torch.float): results of the BLEU metrics
        """
        # Unpack the data batch
        batch_size = batch.batch_size
        img_tensor = batch.img_tensor
        token_tensor = batch.token_tensor
        stop_prob = batch.stop_prob

        # Move to GPU if it is available
        if torch.cuda.is_available():
            img_tensor = img_tensor.cuda()
            token_tensor = token_tensor.cuda()
            stop_prob = stop_prob.cuda()

        # Extract the visual feature
        visual_feature = self.extractor(img_tensor)
        # Predict the distributions of active tags
        tag_probs = self.mlc.get_tag_probs(visual_feature)
        # As per the wingspan project, transpose the visual feature
        visual_feature = visual_feature.transpose(1, 2).contiguous()

        # Generate semantic features from predicted tag probabilities
        semantic_feature = self.tag_generator(tag_probs)

        # Initialization
        stop_losses, word_losses, attention_losses = 0., 0., 0.
        sentence_states = self.sentence_lstm.init_hidden(batch_size)
        visual_aligns, semantic_aligns = [], []

        pred_words = []
        for sentence_index in range(token_tensor.shape[1]):
            # Obtain the topic and pre_stop from the sentence LSTM
            sentence_states, topic, pred_stop, \
                 v_align, s_align, topic_var = self.sentence_lstm(
                    visual_feature,
                    semantic_feature,
                    sentence_states)

            visual_aligns.append(v_align)
            semantic_aligns.append(s_align)

            stop_losses += self.ce_criterion(
                pred_stop, stop_prob[:, sentence_index]
            ).sum()

            # Note the 1 below is to preclude the BOS token
            # Dim of teacher_words: [batch_size, max_word_num],
            # where max_word_num >= max_sentence_len
            teacher_words = token_tensor[:, sentence_index, :].long()

            # Calculate the sentence length for each sample
            mask = teacher_words != self.vocab.pad_token_id
            sentence_len = mask.to(torch.long).sum(1)

            max_sentence_len = sentence_len.max()

            # Generate the words only when at least one sentence in
            # the current batch has content
            if max_sentence_len > 0:
                word_output = self.word_lstm(
                    topic, train=True,
                    inp=teacher_words, sentence_len=sentence_len + 1)

                # Dim of word_output.logits:
                # [batch_size, max_sentence_len + 1, vocab_size]
                word_losses += sequence_sparse_softmax_cross_entropy(
                    labels=teacher_words[:, :max_sentence_len],
                    logits=word_output.logits[:, 1:, :],
                    sequence_length=sentence_len)

                pred_words.append((word_output.sample_id, sentence_len))

        visual_aligns = torch.cat(visual_aligns, -1)
        semantic_aligns = torch.cat(semantic_aligns, -1)
        attention_losses = self.attn_loss(visual_aligns, semantic_aligns)

        train_loss = self.lambda_stop * stop_losses \
                        + self.lambda_word * word_losses \
                        + self.lambda_attn * attention_losses

        pred_paragraph = []
        for j in range(batch_size):
            paragraph = []
            for i in range(token_tensor.shape[1] - 1):
                sen = pred_words[i][0][j]
                len_sen = pred_words[i][1][j]
                sen = sen[1:len_sen + 1].cpu().tolist()
                if len(sen) > 0:
                    sen_tokens = self.vocab.map_ids_to_tokens_py(sen)
                    paragraph.append(' '.join(sen_tokens))

            # print(' '.join(paragraph))
            paragraph = ' '.join(strip_special_tokens(paragraph))
            pred_paragraph.append(paragraph)

        target_paragraph = []
        for i in range(batch_size):
            paragraph = []
            for j in range(token_tensor.shape[1] - 1):
                sen = token_tensor[i, j].long()
                mask = sen != self.vocab.pad_token_id
                len_sen = mask.to(torch.long).sum()
                if len_sen > 0:
                    sen = sen[:len_sen].cpu().tolist()
                    sen_tokens = self.vocab.map_ids_to_tokens_py(sen)
                    paragraph.append(' '.join(sen_tokens))

            paragraph = ' '.join(strip_special_tokens(paragraph))
            target_paragraph.append([paragraph])

        bleu, bleu_1, bleu_2, bleu_3, bleu_4 = corpus_bleu(
            target_paragraph, pred_paragraph, return_all=True)

        return {
            "loss": train_loss,
            "stop_loss": stop_losses,
            "word_loss": word_losses,
            "attention_loss": attention_losses,
            "topic_var": topic_var,
            "bleu": bleu,
            "bleu_1": bleu_1,
            "bleu_2": bleu_2,
            "bleu_3": bleu_3,
            "bleu_4": bleu_4,
        }

    def predict(self, batch):
        r"""Infer the medical reports given the input images.
        """
        # Unpack the data batch
        batch_size = batch.batch_size
        img_tensor = batch.img_tensor
        token_tensor = batch.token_tensor

        # Move to GPU if it is available
        if torch.cuda.is_available():
            img_tensor = img_tensor.cuda()
            token_tensor = token_tensor.cuda()

        # Extract the visual feature
        visual_feature = self.extractor(img_tensor)
        # Predict the distributions of active tags
        tag_probs = self.mlc.get_tag_probs(visual_feature)
        # As per the wingspan project, transpose the visual feature
        visual_feature = visual_feature.transpose(1, 2).contiguous()
        # Generate semantic features from predicted tag probabilities
        semantic_feature = self.tag_generator(tag_probs)
        # semantic_feature = torch.zeros(batch_size, 1, 512).cuda()

        # Initialization
        sentence_states = self.sentence_lstm.init_hidden(batch_size)
        # stopped_mask[i] indiacates whether the topic generation
        # for the i-th sample is stopped (True: stop, False: contine)
        stopped_mask = torch.zeros([batch_size], dtype=torch.bool)

        pred_words = []
        max_sentence_num = 0
        sent_num = 0

        while sent_num < self.max_sent_num:
            # Obtain the topic and pre_stop from the sentence LSTM
            sentence_states, topic, pred_stop, _, _, _ = self.sentence_lstm(
                    visual_feature, semantic_feature, sentence_states)

            stopped = torch.nonzero(pred_stop[:, 0] > 0.5)
            stopped_mask[stopped] = True
            if torch.all(stopped_mask):
                break

            word_output = self.word_lstm(topic, train=False)
            max_sentence_num += 1

            # mask out the results for the sample that already stopped
            word_output.sample_id[stopped_mask] = self.vocab.eos_token_id
            pred_words.append(word_output.sample_id)
            sent_num += 1

        pred_paragraph = []
        for j in range(batch_size):
            paragraph = []
            for i in range(max_sentence_num):
                sen = pred_words[i][j]
                mask = sen == self.vocab.eos_token_id
                try:
                    first_eos_index = mask.nonzero(as_tuple=True)[0][0]
                    if first_eos_index > 0:
                        sen = sen[1:first_eos_index+1].cpu().tolist()
                        sen_tokens = self.vocab.map_ids_to_tokens_py(sen)
                        paragraph.append(' '.join(sen_tokens))
                except IndexError:
                    continue

            paragraph = ' '.join(strip_special_tokens(paragraph))
            pred_paragraph.append(paragraph)

        target_paragraph = []
        for i in range(batch_size):
            paragraph = []
            for j in range(token_tensor.shape[1] - 1):
                sen = token_tensor[i, j].long()
                mask = sen != self.vocab.pad_token_id
                len_sen = mask.to(torch.long).sum()
                if len_sen > 0:
                    sen = sen[:len_sen].cpu().tolist()
                    sen_tokens = self.vocab.map_ids_to_tokens_py(sen)
                    paragraph.append(' '.join(sen_tokens))

            paragraph = ' '.join(strip_special_tokens(paragraph))
            target_paragraph.append([paragraph])

        bleu, bleu_1, bleu_2, bleu_3, bleu_4 = corpus_bleu(
            target_paragraph, pred_paragraph, return_all=True)

        return {
            'bleu': bleu,
            'bleu_1': bleu_1,
            'bleu_2': bleu_2,
            'bleu_3': bleu_3,
            'bleu_4': bleu_4,
        }

    @staticmethod
    def default_hparams() -> Dict[str, Any]:
        r"""Returns a dictionary of hyperparameters with default values.

        Returns: (dict) default hyperparameters

        """
        return {
                "mlc": None,
                "tag_generator": None,
                "sentence_lstm": None,
                "word_lstm": None,
                "lambda_stop": 1.,
                "lambda_word": 1.,
                "lambda_attn": 1.,
                "max_sent_num": 14,
                "visual_weights": None,
                "pathologies": None,
                "vocab_path": None,
                "train_visual": False,
        }
