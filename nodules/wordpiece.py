# adapted from https://huggingface.co/learn/nlp-course/en/chapter6/6

from collections import defaultdict


class WordPiece:
    def __init__(self, vocab_size, additional_tokens=None, unknown_token='[UNK]'):
        self.word_freqs = None  # word freqs to fit on
        self.additional_tokens = additional_tokens + [unknown_token] if (additional_tokens
                                                                         is not None) else [unknown_token]
        self.unknown_token = unknown_token
        self.vocab_size = vocab_size
        self.vocab = None  # vocab
        self.splits = None  # word : split word

    def _create_alphabet(self):
        """
        Creates alphabet of letters, possibly with ## in front to indicate they are
        not the starting letter
        :return: none, populates vocab and splits
        """
        alphabet = []
        for word in self.word_freqs.keys():
            alphabet.append(word[0])
            for letter in word[1:]:
                if f"##{letter}" not in alphabet:
                    alphabet.append(f"##{letter}")

        vocab = self.additional_tokens + alphabet
        splits = {
            word: [c if i == 0 else f"##{c}" for i, c in enumerate(word)]
            for word in self.word_freqs.keys()
        }
        self.vocab, self.splits = set(vocab), splits

    def _compute_pair_scores(self):
        """
        I THINK pair scores are
        pair_frequency / token1_freq * token2_freq
        :return:
        """
        letter_freqs = defaultdict(int)
        pair_freqs = defaultdict(int)

        for word, freq in self.word_freqs.items():
            split = self.splits[word]
            if len(split) == 1:
                letter_freqs[split[0]] += freq
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                letter_freqs[split[i]] += freq
                pair_freqs[pair] += freq
            letter_freqs[split[-1]] += freq

        scores = {
            pair: freq / (letter_freqs[pair[0]] * letter_freqs[pair[1]])
            for pair, freq in pair_freqs.items()
        }
        return scores

    def _merge_pair(self, a, b, splits):
        """
        Merges a pair inside the splits
        """
        for word in self.word_freqs:
            split = splits[word]
            if len(split) == 1:
                continue
            i = 0
            while i < len(split) - 1:
                if split[i] == a and split[i + 1] == b:
                    merge = a + b[2:] if b.startswith('##') else a + b
                    split = split[:i] + [merge] + split[i + 2:]
                else:
                    i += 1
            splits[word] = split
        return splits

    def _get_tokens(self):
        """
        Gets tokens, assuming splits is populated
        :return:
        """
        while len(self.vocab) < self.vocab_size:
            scores = self._compute_pair_scores()
            best_pair, max_score = "", None
            for pair, score in scores.items():
                if max_score is None or max_score < score:
                    best_pair = pair
                    max_score = score
            self.splits = self._merge_pair(*best_pair, self.splits)
            new_token = (
                best_pair[0] + best_pair[1][2:]
                if best_pair[1].startswith('##')
                else best_pair[0] + best_pair[1]
            )
            self.vocab.add(new_token)

    def _encode_word(self, word):
        """
        Basically, continuously finds the longest token possible and adds that
        to a list of tokens. Returns unknown token if there's nothing
        :param word: the word to encode
        :return: a list of tokens
        """
        tokens = []
        while len(word) > 0:
            i = len(word)
            while i > 0 and word[:i] not in self.vocab:
                i -= 1
            if i == 0:
                return [self.unknown_token]
            tokens.append(word[:i])
            word = word[i:]
            if len(word) > 0:
                word = f"##{word}"
        return tokens

    def fit(self, word_freqs):
        self.word_freqs = word_freqs
        self._create_alphabet()
        self._get_tokens()

    def tokenize(self, text):
        """
        tokenizes text
        :param text: a list of words
        :return: a list of tokens
        """
        encoded_words = [self._encode_word(word) for word in text]
        return sum(encoded_words, [])
