"""Generators for characters and surrounding context."""

import string
from random import choice, randint, random, uniform

TINY_PUNCT = '.-,;'
OTHER_PUNCT = """$%*()<=>+/:#&"""
CHARS = sorted(string.digits + string.ascii_uppercase + TINY_PUNCT + OTHER_PUNCT)
CHAR_TO_IDX = {c: i for i, c in enumerate(CHARS)}
IDX_TO_CHAR = {v: k for k, v in CHAR_TO_IDX.items()}

ALNUM = string.digits + string.ascii_uppercase


def get_chars(chars, *, left=1, right=1):
    """Return chars from the string."""
    word = f' {chars.upper()} '
    idx = randint(1, len(word) - 2)
    start = idx - left
    end = idx + right + 1
    return word[start:end]


def float_chars(*, low=-2.0, high=2.0, precision=8, left=1, right=1):
    """Generate characters for floating point digits."""
    num = uniform(low, high)
    leader = choice(':=') if random() < 0.05 else ''
    word = f'{leader}{num:0.{precision}f}'
    return get_chars(word, left=left, right=right)


def single_chars(*, left=1, right=1):
    """Generate words and pick some characters out of them."""
    word = choice(CHARS)
    return get_chars(word, left=left, right=right)


def word_chars(*, left=1, right=1, min_len=1, max_len=10):
    """Generate words and pick some characters out of them."""
    chars = ALNUM if random() < 0.1 else string.ascii_uppercase
    word = [choice(chars) for _ in range(randint(min_len, max_len))]
    word = ''.join(word)
    leader = choice(':=') if random() < 0.05 else ''
    trailer = choice(':=,;') if random() < 0.05 else ''
    word = f'{leader}{word}{trailer}'
    return get_chars(word, left=left, right=right)


def int_chars(*, low=1, high=100, left=1, right=1):
    """Generate characters for floating point digits."""
    num = randint(low, high)
    leader = choice(':=') if random() < 0.05 else ''
    trailer = choice('%:,;') if random() < 0.05 else ''
    word = f'{leader}{num}{trailer}'
    return get_chars(word, left=left, right=right)
