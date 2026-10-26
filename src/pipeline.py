from llm_sdk import Small_LLM_Model
from typing import Union
from .file_handler import Function, CallResult
from .utils import PromptBuilder, GenerationLogger, to_type
import json


class TrieNode:
    """
    Represents a single node in a token prefix tree (trie).

    Attributes:
        child: Map of token ID to child TrieNode instances.
        name: Terminal identifier name if node forms a complete sequence.
    """
    def __init__(self) -> None:
        """Initialize an empty trie node."""
        self.child: dict[int, TrieNode] = {}
        self.name: str | None = None


class Trie:
    """
    Prefix tree (trie) data structure for tracking allowed token sequences.

    Attributes:
        root: Root node of the prefix tree.
    """
    def __init__(self) -> None:
        """Initialize the trie with a root node."""
        self.root = TrieNode()

    def insert(self, token_ids: list[int], name: str) -> None:
        """
        Insert a sequence of token IDs associated with a string identifier.

        Args:
            token_ids: Sequence of integer token IDs forming the key.
            name: String payload stored at the final sequence node.
        """
        current = self.root

        for token in token_ids:
            if token not in current.child:
                current.child[token] = TrieNode()
            current = current.child[token]

        current.name = name

    def _get_node(self, prefix: list[int]) -> TrieNode | None:
        """
        Retrieve the node corresponding to a prefix sequence.

        Args:
            prefix: Sequence of token IDs to traverse.
        Returns:
            The terminal TrieNode if prefix exists, or None.
        """
        current = self.root
        for token in prefix:
            if token not in current.child:
                return None
            current = current.child[token]
        return current

    def allowed_tokens(self, prefix: list[int]) -> set[int]:
        """
        Find valid next token IDs given a generated prefix.

        Args:
            prefix: Sequence of previously generated token IDs.
        Returns:
            Set of allowable next token IDs leading to valid sequence paths.
        """
        node = self._get_node(prefix)
        if node is None:
            return set()
        return set(node.child.keys())

    def get_name(self, prefix: list[int]) -> str | None:
        """
        Retrieve the associated name string for a complete token sequence.

        Args:
            prefix: Sequence of token IDs to check.
        Returns:
            Associated name string if path is valid terminal node, else None.
        """
        node = self._get_node(prefix)

        if node is None:
            return None
        return node.name

    def is_complete(self, prefix: list[int]) -> bool:
        """
        Check whether a token sequence forms a completed string in trie.

        Args:
            prefix: Sequence of token IDs to test.
        Returns:
            True if sequence ends on a named terminal node, False otherwise.
        """
        return self.get_name(prefix) is not None


def mask_logits(logits: list[float], allowed: set[int]) -> list[float]:
    """
    Mask probability logits by setting disallowed tokens to -infinity.

    Args:
        logits: List of raw output logits from model prediction.
        allowed: Set of token IDs permitted for selection.
    Returns:
        A list of logits with disallowed token indices zeroed out via -inf.
    """
    for token in range(len(logits)):
        if token not in allowed:
            logits[token] = float("-inf")
    return logits


class VocabConstraints:
    """
    Builds token mask sets based on allowed and forbidden character rules.

    Attributes:
        vocab: Dictionary mapping token text representations to token IDs.
        str_allowed: Permitted token set for general string parameters.
        num_allowed: Permitted token set for float/number parameter values.
        int_allowed: Permitted token set for integer parameter values.
        gen_configs: Configuration tuples for each supported parameter type.
    """
    def __init__(self, llm: Small_LLM_Model) -> None:
        """
        Initialize vocabulary constraint sets from an LLM model instance.

        Args:
            llm: Small_LLM_Model instance used
                for decoding and vocabulary access.
        """
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
        """
        Load vocabulary mapping from the underlying model directory.

        Args:
            llm: Language model instance providing the vocabulary file path.
        Returns:
            Dictionary mapping string token symbols to integer token IDs.
        """
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
        """
        Create token ID sets filtered by allowable
        or forbidden character rules.

        Args:
            llm: Language model instance used to decode token IDs into text.
            to_allow: String containing chars that tokens MUST consist of.
            to_forbid: String containing chars that tokens MUST NOT contain.
        Returns:
            Set of allowed token ID integers.
        """
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
    """
    Manages token generation driven by trie paths or vocab mask constraints.

    Attributes:
        llm: Language model instance generating tokens.
        logger: Logger instance recording token outputs.
        input_ids: Running history of prompt and generated token IDs.
        max_tokens: Limit on continuous generation steps
                    during masked decoding.
    """
    def __init__(
        self,
        llm: Small_LLM_Model,
        logger: GenerationLogger,
        max_tokens: int = 15
    ) -> None:
        """
        Initialize the decoder with language model and logging references.

        Args:
            llm: Language model used for logit evaluation and encoding.
            logger: GenerationLogger tracking generated text output.
            max_tokens: Maximum token threshold per mask step.
                        Defaults to 15.
        """
        self.llm = llm
        self.logger = logger
        self.input_ids: list[int] = []
        self.max_tokens = max_tokens

    def add_to_trie(self, trie: Trie, to_encode: str) -> None:
        """
        Encode text and insert resulting token IDs into a specified trie.

        Args:
            trie: Prefix tree instance to populate.
            to_encode: Target string to tokenize and store.
        """
        token_ids = self.llm.encode(to_encode)[0].tolist()
        trie.insert(token_ids, to_encode)

    def inject(self, to_encode: str) -> None:
        """
        Append encoded string tokens directly into the decoder input context.

        Args:
            to_encode: Literal text fragment to add to decoder context.
        """
        token_ids: list[int] = self.llm.encode(to_encode)[0].tolist()
        self.input_ids.extend(token_ids)
        self.logger.token(self.llm.decode(token_ids))

    def generate_from_trie(self, trie: Trie) -> list[int]:
        """
        Generate token sequences strictly guided by valid paths in a trie.

        Args:
            trie: Constraints trie specifying allowed token branches.
        Returns:
            List of generated token IDs forming a complete path in the trie.
        """
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
        """
        Generate tokens filtered by vocab masks until a stop char is hit.

        Args:
            allowed: Permitted set of token IDs for iteration steps.
            stop_chars: Tuple of chars that trigger termination when found.
        Returns:
            List of generated token IDs prior to encountering a stop char.
        """
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
            if stop_index != -1:
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
    """
    Orchestrates schema-constrained generation
    of structured JSON function calls.

    Attributes:
        functions: Available function specifications.
        llm: Underlying language model wrapper.
        logger: Token output logger.
        constraints: Vocabulary constraints manager.
        decoder: Constrained decoding controller.
        func_trie: Trie containing valid function names.
        param_tries: Cache mapping (function, param) keys to parameter tries.
        bool_trie: Trie populated with boolean literals ('true', 'false').
        curr_func: Active Function schema selected during generation.
        remaining_params: List of unparsed parameters
                          required for current function.
    """
    def __init__(
        self,
        functions: list[Function],
        llm: Small_LLM_Model
    ) -> None:
        """
        Initialize generator with candidate function schemas
        and language model.

        Args:
            functions: List of Function schemas available for invocation.
            llm: Language model instance powering token generation.
        """
        self.functions = functions
        self.llm = llm
        self.logger = GenerationLogger()
        self.constraints = VocabConstraints(llm)
        self.decoder = ConstrainedDecoder(llm, self.logger)
        self.func_trie: Trie = self._gen_func_trie(self.functions)
        self.param_tries: dict[str, dict[str, Trie]] = {}
        self.bool_trie = Trie()
        for val in ("false", "true"):
            self.decoder.add_to_trie(self.bool_trie, val)
        self.curr_func: Function
        self.remaining_params: list[str] = []

    def _gen_func_trie(self, functions: list[Function]) -> Trie:
        """
        Construct a prefix trie containing all available function names.

        Args:
            functions: Schemas representing valid callable functions.
        Returns:
            Populated Trie instance containing function name paths.
        """
        func_trie = Trie()
        for func in functions:
            self.decoder.add_to_trie(func_trie, func.name)
        return func_trie

    def _get_param_trie(self, func_name: str, param_name: str) -> Trie:
        """
        Fetch or initialize a parameter name trie for a specific function.

        Args:
            func_name: Name of target function schema.
            param_name: Target parameter key name.
        Returns:
            Trie instance encoding the parameter name sequence.
        """
        if func_name not in self.param_tries:
            self.param_tries[func_name] = {}
        if param_name not in self.param_tries[func_name]:
            param_trie = Trie()
            self.decoder.add_to_trie(param_trie, param_name)
            self.param_tries[func_name][param_name] = param_trie
        return self.param_tries[func_name][param_name]

    def _gen_func_name(self) -> str:
        """
        Generate a valid function name guided by the function trie.

        Returns:
            Decoded string representing chosen function name.
        """
        func_name_ids = self.decoder.generate_from_trie(self.func_trie)
        return self.llm.decode(func_name_ids)

    def _gen_param_name(self) -> str:
        """
        Generate the next required parameter name for current function call.

        Returns:
            Decoded string representing chosen parameter name.
        """
        next_param = self.remaining_params[0]
        param_trie = self._get_param_trie(self.curr_func.name, next_param)
        param_name_ids = self.decoder.generate_from_trie(param_trie)
        self.remaining_params.remove(next_param)
        return self.llm.decode(param_name_ids)

    def _gen_param_value(self, p_type: str) -> str:
        """
        Generate a parameter value constrained by data type specifications.

        Args:
            p_type: Data type requirement
                    ('boolean', 'string', 'number', 'integer').
        Returns:
            Decoded parameter value as raw string text.
        """
        if p_type == "boolean":
            param_val_ids = self.decoder.generate_from_trie(self.bool_trie)
            return self.llm.decode(param_val_ids)
        else:
            allowed, stop_chars = self.constraints.gen_configs[p_type]

            param_val_ids = self.decoder.generate_from_mask(
                allowed,
                stop_chars
            )
            return self.llm.decode(param_val_ids)

    def run(self, user_prompt: str) -> CallResult:
        """
        Run constrained generation to transform a user prompt
        into a CallResult.

        Args:
            user_prompt: Free-form text prompt submitted by user.
        Returns:
            CallResult instance containing
            selected function and parameter arguments.
        """
        prompter = PromptBuilder()
        prompt = prompter. make_prompt(self.functions, user_prompt)
        self.decoder.input_ids = self.llm.encode(prompt)[0].tolist()
        self.logger.prompt(user_prompt)
        self.decoder.inject('{\n  "name": "')
        func_name = self._gen_func_name()
        params: dict[str, Union[str, int, float, bool]] = {}

        if func_name == "fn_none_found":
            self.decoder.inject('",\n  "parameters": {}\n}')
            return CallResult(
                prompt=user_prompt,
                name=func_name,
                parameters=params
            )
        self.decoder.inject('",\n  "parameters": {\n')

        for func in self.functions:
            if func_name == func.name:
                self.curr_func = func
                self.remaining_params = list(func.parameters.keys())

        while self.remaining_params:
            self.decoder.inject('    "')

            p_name = self._gen_param_name()
            p_type = self.curr_func.parameters[p_name].type
            is_num = p_type in {'number', 'integer'}
            self.decoder.inject('": ' if is_num else '": "')

            p_value = self._gen_param_value(p_type)
            converted_val: str | int | float | bool = to_type(p_type, p_value)
            params[p_name] = converted_val

            if self.remaining_params:
                self.decoder.inject(',\n' if is_num else '",\n')
            else:
                self.decoder.inject('\n  }\n}' if is_num else '"\n  }\n}')

        return CallResult(
            prompt=user_prompt,
            name=func_name,
            parameters=params
        )
