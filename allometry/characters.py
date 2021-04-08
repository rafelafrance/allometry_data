"""Generators for characters and surrounding context."""

from random import choice, random, randint, randrange
from random_word import RandomWords
import string

rand_words = RandomWords()


def get_chars(chars, *, left=1, right=1):
    """Return chars from the string."""
    word = f' {chars.upper()} '
    idx = randint(1, len(word) - 2)
    start = idx - left
    end = idx + right + 1
    return list(word[start:end])


def float_chars(*, left=1, right=1, scale=2.0, neg_bias=0.5):
    """Generate characters for floating point digits."""
    num = random() * scale
    num = -num if random() < neg_bias else num
    word = f'{num:0.8f}'
    return get_chars(word, left=left, right=right)


def word_chars(*, left=1, right=1):
    """Generate words and pick some characters out of them."""
    if random() < 0.1:
        idx = randrange(26)
        word = string.ascii_uppercase[idx]
    else:
        word = rand_words.get_random_word()
    return get_chars(word, left=left, right=right)


def int_chars(*, left=1, right=1, low=1, high=100):
    """Generate characters for floating point digits."""
    num = randint(low, high)
    leader = choice(':=') if random() < 0.05 else ''
    trailer = choice('%:,;') if random() < 0.05 else ''
    word = f' {leader}{num}{trailer} '
    return get_chars(word, left=left, right=right)
