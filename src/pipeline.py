from llm_sdk import Small_LLM_Model
from typing import Union
from .file_handler import Function, CallResult
from .utils import PromptBuilder, GenerationLogger, to_type
import json


class TrieNode:
    """
    A node in a tree (trie) that stores token ID pathing and terminal node
    """
    def __init__(self) -> None:
        self.child: dict[int, TrieNode] = {}
        self.name: str | None = None


class Trie:
    """A node tree that stores and matches sequences of token IDs"""
    def __init__(self) -> None:
        self.root = TrieNode()

    def insert(self, token_ids: list[int], name: str) -> None:
        """Puts a token sequence in the trie"""
        current = self.root

        for token in token_ids:
            if token not in current.child:
                current.child[token] = TrieNode()
            current = current.child[token]

        current.name = name

    def _get_node(self, prefix: list[int]) -> TrieNode | None:
        """returns a trie node corresponding to a tokne prefix"""
        current = self.root
        for token in prefix:
            if token not in current.child:
                return None
            current = current.child[token]
        return current

    def allowed_tokens(self, prefix: list[int]) -> set[int]:
        """returns next valid token IDs for a prefix"""
        node = self._get_node(prefix)
        if node is None:
            return set()
        return set(node.child.keys())

    def get_name(self, prefix: list[int]) -> str | None:
        """returns value (name) associated with a token sequence"""
        node = self._get_node(prefix)

        if node is None:
            return None
        return node.name

    def is_complete(self, prefix: list[int]) -> bool:
        """checks if a token sequence is fully in the trie"""
        return self.get_name(prefix) is not None


def mask_logits(logits: list[float], allowed: set[int]) -> list[float]:
    """replaces logits of disallowed tokens with negative infinity."""
    for token in range(len(logits)):
        if token not in allowed:
            logits[token] = float("-inf")
    return logits


class VocabConstraints:
    """builds token masks to constrain text generation"""
    def __init__(self, llm: Small_LLM_Model) -> None:
        self.vocab: dict[str, int] = self._get_vocab(llm)
        self.str_allowed: set[int] = self._make_mask(
            llm, to_forbid="`#$"
        )
        self.num_allowed: set[int] = self._make_mask(
            llm, to_allow=",\n0123456789-"
        )
        self.int_allowed: set[int] = self._make_mask(
            llm, to_allow=",\n0123456789-."
        )
        self.gen_configs: dict[str, tuple[set[int], tuple[str, ...]]] = {
            "string": (self.str_allowed, ('"',)),
            "number": (self.num_allowed, (',', '\n')),
            "integer": (self.int_allowed, (',', '\n')),
        }


    def _get_vocab(self, llm: Small_LLM_Model) -> dict[str, int]:
        """loads model vocab from disk"""
        path = llm.get_path_to_vocab_file()
        with open(path, 'r', encoding='utf-8') as f:
            vocab: dict[str, int] = json.load(f)
        return vocab

    def _make_mask(
            self,
            llm: Small_LLM_Model,
            *,
            to_allow: str = "",
            to_forbid: str = "",
            ) -> set[int]:
        """creates a token mask based on allowed or disallowed characters"""
        allowed = []
        allow_set = set(to_allow)
        forbid_set = set(to_forbid)

        for token in self.vocab.values():
            token_text = llm.decode([token])
            if allow_set:
                if all(c in allow_set for c in token_text):
                    allowed.append(token)
            elif forbid_set:
                if all(c not in token_text for c in forbid_set):
                    allowed.append(token)
        return set(allowed)


class ConstrainedDecoder:
    """generates tokens while enforcing trie or vocab constraints"""
    def __init__(
        self,
        llm: Small_LLM_Model,
        logger: GenerationLogger,
        max_tokens: int = 15
    ) -> None:
        self.llm = llm
        self.logger = logger
        self.input_ids: list[int] = []
        self.max_tokens = max_tokens

    def add_to_trie(self, trie: Trie, to_encode: str) -> None:
        """encodes a string a and puts it in a trie"""
        token_ids = self.llm.encode(to_encode)[0].tolist()
        trie.insert(token_ids, to_encode)

    def inject(self, to_encode: str) -> None:
        """appends encoded text to decoder input"""
        token_ids: list[int] = self.llm.encode(to_encode)[0].tolist()
        self.input_ids.extend(token_ids)
        self.logger.token(self.llm.decode(token_ids))

    def generate_from_trie(self, trie: Trie) -> list[int]:
        """generates tokens constrained by a trie"""
        generated: list[int] = []

        while not trie.is_complete(generated):
            allowed = trie.allowed_tokens(generated)
            logits = self.llm.get_logits_from_input_ids(self.input_ids)
            masked_logits = mask_logits(logits, allowed)
            next_token = masked_logits.index(max(masked_logits))
            self.input_ids.append(next_token)
            generated.append(next_token)
            self.logger.token(self.llm.decode([next_token]))
        return generated

    def generate_from_mask(
        self,
        allowed: set[int],
        stop_chars: tuple[str, ...]
    ) -> list[int]:
        """generates tokens using a vocab mask until a stop char"""
        generated: list[int] = []

        while len(generated) < self.max_tokens:
            logits = self.llm.get_logits_from_input_ids(self.input_ids)
            masked_logits = mask_logits(logits, allowed)
            next_token = masked_logits.index(max(masked_logits))

            token_val = self.llm.decode([next_token])
            stop_index = min(
                (token_val.index(c) for c in stop_chars if c in token_val),
                default=-1
            )
            if stop_index != 1:
                prefix: str = token_val[:stop_index]
                if prefix:
                    prefix_ids: list[int] = self.llm.encode(prefix)[0].tolist()
                    self.input_ids.extend(prefix_ids)
                    generated.extend(prefix_ids)
                    self.logger.token(prefix)
                break
            self.input_ids.append(next_token)
            generated.append(next_token)
            self.logger.token(token_val)
        return generated


class FunctionCallGenerator:
    """generates function calls using constrained decoding"""
    def __init__(
        self,
        functions: list[Function],
        llm: Small_LLM_Model
    ) -> None:
        self.functions = function
        self.llm = llm
