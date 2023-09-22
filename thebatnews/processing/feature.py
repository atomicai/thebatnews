import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizer, RobertaTokenizer

logger = logging.getLogger(__name__)

#: Special characters used by the different tokenizers to indicate start of word / whitespace
SPECIAL_TOKENIZER_CHARS = r"^(##|Ġ|▁)"


def truncate_sequences(
    seq_a: list,
    seq_b: Optional[list],
    tokenizer: AutoTokenizer,
    max_seq_len: int,
    truncation_strategy: str = "longest_first",
    with_special_tokens: bool = True,
    stride: int = 0,
) -> Tuple[List[Any], Optional[List[Any]], List[Any]]:
    """
    Reduces a single sequence or a pair of sequences to a maximum sequence length.
    The sequences can contain tokens or any other elements (offsets, masks ...).
    If `with_special_tokens` is enabled, it'll remove some additional tokens to have exactly
    enough space for later adding special tokens (CLS, SEP etc.)

    Supported truncation strategies:

    - longest_first: (default) Iteratively reduce the inputs sequence until the input is under
        max_length starting from the longest one at each token (when there is a pair of input sequences).
        Overflowing tokens only contains overflow from the first sequence.
    - only_first: Only truncate the first sequence. raise an error if the first sequence is
        shorter or equal to than num_tokens_to_remove.
    - only_second: Only truncate the second sequence
    - do_not_truncate: Does not truncate (raise an error if the input sequence is longer than max_length)

    :param seq_a: First sequence of tokens/offsets/...
    :param seq_b: Optional second sequence of tokens/offsets/...
    :param tokenizer: Tokenizer (e.g. from get_tokenizer))
    :param max_seq_len:
    :param truncation_strategy: how the sequence(s) should be truncated down.
        Default: "longest_first" (see above for other options).
    :param with_special_tokens: If true, it'll remove some additional tokens to have exactly enough space
        for later adding special tokens (CLS, SEP etc.)
    :param stride: optional stride of the window during truncation
    :return: truncated seq_a, truncated seq_b, overflowing tokens
    """
    pair = seq_b is not None
    len_a = len(seq_a)
    len_b = len(seq_b) if seq_b is not None else 0
    num_special_tokens = tokenizer.num_special_tokens_to_add(pair=pair) if with_special_tokens else 0
    total_len = len_a + len_b + num_special_tokens
    overflowing_tokens = []

    if max_seq_len and total_len > max_seq_len:
        seq_a, seq_b, overflowing_tokens = tokenizer.truncate_sequences(
            seq_a,
            pair_ids=seq_b,
            num_tokens_to_remove=total_len - max_seq_len,
            truncation_strategy=truncation_strategy,
            stride=stride,
        )
    return (seq_a, seq_b, overflowing_tokens)


#
# FIXME this is a relic from FARM. If there's the occasion, remove it!
#
def tokenize_with_metadata(text: str, tokenizer: PreTrainedTokenizer) -> Dict[str, Any]:
    """
    Performing tokenization while storing some important metadata for each token:

    * offsets: (int) Character index where the token begins in the original text
    * start_of_word: (bool) If the token is the start of a word. Particularly helpful for NER and QA tasks.

    We do this by first doing whitespace tokenization and then applying the model specific tokenizer to each "word".

    .. note::  We don't assume to preserve exact whitespaces in the tokens!
               This means: tabs, new lines, multiple whitespace etc will all resolve to a single " ".
               This doesn't make a difference for BERT + XLNet but it does for RoBERTa.
               For RoBERTa it has the positive effect of a shorter sequence length, but some information about whitespace
               type is lost which might be helpful for certain NLP tasks ( e.g tab for tables).

    :param text: Text to tokenize
    :param tokenizer: Tokenizer (e.g. from get_tokenizer))
    :return: Dictionary with "tokens", "offsets" and "start_of_word"
    """
    # normalize all other whitespace characters to " "
    # Note: using text.split() directly would destroy the offset,
    # since \n\n\n would be treated similarly as a single \n
    text = re.sub(r"\s", " ", text)

    words: Union[List[str], np.ndarray] = []
    word_offsets: Union[List[int], np.ndarray] = []
    start_of_word: List[Union[int, bool]] = []

    # Fast Tokenizers return offsets, so we don't need to calculate them ourselves
    if tokenizer.is_fast:
        # tokenized = tokenizer(text, return_offsets_mapping=True, return_special_tokens_mask=True)
        tokenized = tokenizer(text, return_offsets_mapping=True, return_special_tokens_mask=True)

        tokens = tokenized["input_ids"]
        offsets = np.array([x[0] for x in tokenized["offset_mapping"]])
        # offsets2 = [x[0] for x in tokenized2["offset_mapping"]]
        words = np.array(tokenized.encodings[0].words)

        # TODO check for validity for all tokenizer and special token types
        words[0] = -1
        words[-1] = words[-2]
        words += 1
        start_of_word = [0] + list(np.ediff1d(words))
        return {"tokens": tokens, "offsets": offsets, "start_of_word": start_of_word}

    # split text into "words" (here: simple whitespace tokenizer).
    words = text.split(" ")
    cumulated = 0
    for word in words:
        word_offsets.append(cumulated)  # type: ignore [union-attr]
        cumulated += len(word) + 1  # 1 because we so far have whitespace tokenizer

    # split "words" into "subword tokens"
    tokens, offsets, start_of_word = _words_to_tokens(words, word_offsets, tokenizer)  # type: ignore
    return {"tokens": tokens, "offsets": offsets, "start_of_word": start_of_word}


# Note: only used by tokenize_with_metadata()
def _words_to_tokens(
    words: List[str], word_offsets: List[int], tokenizer: PreTrainedTokenizer
) -> Tuple[List[str], List[int], List[bool]]:
    """
    Tokenize "words" into subword tokens while keeping track of offsets and if a token is the start of a word.
    :param words: list of words.
    :param word_offsets: Character indices where each word begins in the original text
    :param tokenizer: Tokenizer (e.g. from get_tokenizer))
    :return: Tuple of (tokens, offsets, start_of_word)
    """
    tokens: List[str] = []
    token_offsets: List[int] = []
    start_of_word: List[bool] = []
    index = 0
    for index, (word, word_offset) in enumerate(zip(words, word_offsets)):
        if index % 500000 == 0:
            logger.info(index)
        # Get (subword) tokens of single word.

        # empty / pure whitespace
        if len(word) == 0:
            continue
        # For the first word of a text: we just call the regular tokenize function.
        # For later words: we need to call it with add_prefix_space=True to get the same results with roberta / gpt2 tokenizer
        # see discussion here. https://github.com/huggingface/transformers/issues/1196
        if len(tokens) == 0:
            tokens_word = tokenizer.tokenize(word)
        else:
            if type(tokenizer) == RobertaTokenizer:
                tokens_word = tokenizer.tokenize(word, add_prefix_space=True)
            else:
                tokens_word = tokenizer.tokenize(word)
        # Sometimes the tokenizer returns no tokens
        if len(tokens_word) == 0:
            continue
        tokens += tokens_word

        # get global offset for each token in word + save marker for first tokens of a word
        first_token = True
        for token in tokens_word:
            token_offsets.append(word_offset)
            # Depending on the tokenizer type special chars are added to distinguish tokens with preceding
            # whitespace (=> "start of a word"). We need to get rid of these to calculate the original length of the token
            original_token = re.sub(SPECIAL_TOKENIZER_CHARS, "", token)
            # Don't use length of unk token for offset calculation
            if original_token == tokenizer.special_tokens_map["unk_token"]:
                word_offset += 1
            else:
                word_offset += len(original_token)
            if first_token:
                start_of_word.append(True)
                first_token = False
            else:
                start_of_word.append(False)

    return tokens, token_offsets, start_of_word


__all__ = ["truncate_sequences", "tokenize_with_metadata"]
