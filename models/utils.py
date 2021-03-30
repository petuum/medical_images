import torch
from texar.torch.modules.decoders.decoder_helpers \
    import GreedyEmbeddingHelper


class InferenceHelper(GreedyEmbeddingHelper):
    r"""A helper for use during inference.

    Uses the argmax of the output (treated as logits) and passes the
    result through an token embedder to get the next input when the
    time > 1.

    Note that the first input (time = 0) is the topic vector, and the
    second input (time = 1) is the embedding of the BOS token.
    Args:
        time_major (bool):  Whether the tensors in ``inputs`` are time major.
            If `False` (default), they are assumed to be batch major.
        token_embedder (torch.nn.Embedding): Embedding function to embed
            the sampled tokens.
    """
    def __init__(
        self, *args,
        time_major=False,
        token_embedder=None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self._time_major = time_major
        self.token_embedder = token_embedder
        self._inputs = None

    def initialize(self, embedding_fn, inputs, sequence_length):
        r"""Initialize the current batch.

        Args:
            embedding_fn: embedding function of the decoder. We do
                not need it in our case.
            inputs: Input tensors that contains topics and embedding
                of the BOS token.
            sequence_length: An int32 vector tensor. We do
                not need it in our case.

        Returns:
            ``(initial_finished, initial_inputs)``.

        Raises:
            ValueError: if :attr:`inputs` is None.
        """
        del embedding_fn, sequence_length
        if inputs is None:
            raise ValueError("`inputs` cannot be None for InferenceHelper")
        inputs: torch.Tensor

        if not self._time_major:
            inputs = inputs.transpose(0, 1)  # make inputs time major
        self._inputs = inputs
        finished = torch.zeros(self._start_tokens.shape[0], dtype=torch.bool)
        return (finished.cuda(), self._inputs[0])

    def next_inputs(self, embedding_fn,
                    time: int, outputs: torch.Tensor,
                    sample_ids: torch.LongTensor):
        r"""next_inputs are set to the embedding of the start tokens
        when next_time == 1. When next_time == 1, it returns the same
        (finished, next_inputs) as the GreedyEmbeddingHelper

        Returns ``(finished, next_inputs)``.
        """
        del embedding_fn, outputs  # unused by next_inputs_fn
        next_time = time + 1
        if next_time == 1:
            finished = torch.zeros(sample_ids.shape[0], dtype=torch.bool)
            next_inputs = self._inputs[next_time]
        else:
            finished = (sample_ids == self._end_token)
            all_finished = torch.all(finished).item()
            embeddings = self.token_embedder(sample_ids)
            next_inputs = (embeddings if not all_finished else self._inputs[1])

        return (finished.cuda(), next_inputs)
