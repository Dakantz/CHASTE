from math import inf

from llama_cpp import Llama, LlamaGrammar
import numpy as np
from dataclasses import dataclass, field

from torch import add
from .grammar import GBNF_PARSER, test_against_grammar


@dataclass
class BeamSearchConfig:
    top_k: int = 3
    max_depth: int = 4
    end_token: str | None = "\n"
    skip_tokens: int = 1
    oversample: int = 8129
    k_progress: list[int] | None = field(
        default_factory=lambda: [4, 2, -1]
    )  # k for depth, -1 means to take top until end_token
    length_cost: float = 0


class BeamSearchNode:
    def __init__(
        self,
        model: Llama,
        grammar: LlamaGrammar,
        cfg: BeamSearchConfig = BeamSearchConfig(),
        lark_grammar=None,
        depth=0,
        input_tokens: list[int] = None,
    ):
        self.children: list[BeamSearchNode] = []
        self.model = model
        self.grammar = grammar
        self.token: int = -1
        self.token_str: str = None
        self.log_p = 1
        self.cfg = cfg
        self.depth = depth
        self.input_tokens = input_tokens
        self.new_tokens: list[int] = []
        self.new_str: str = ""
        if lark_grammar is None:
            # print("Parsing grammar with lark...", grammar._grammar)
            self.lark_grammar = GBNF_PARSER.parse(grammar._grammar)
            # print("Parsed grammar:\n", self.lark_grammar.pretty())
        else:
            self.lark_grammar = lark_grammar

    def eos_token_id(self):
        return int(self.model.metadata.get("tokenizer.ggml.eos_token_id", -1))

    def bos_token_id(self):
        return int(self.model.metadata.get("tokenizer.ggml.bos_token_id", -1))

    def is_end(self):
        if self.cfg.end_token is not None and self.new_str.endswith(self.cfg.end_token):
            return True
        if self.token == int(
            self.model.metadata.get("tokenizer.ggml.eos_token_id", -1)
        ):
            return True
        if self.token == int(
            self.model.metadata.get("tokenizer.ggml.bos_token_id", -1)
        ):
            return True
        # if len(self.token_str) == 0:
        #     return True
        return False

    def filter_logits(self, logits: dict[int, float]):
        filtered_logits: dict[int, float] = {}
        for t, log_p in logits.items():
            try:
                new_string = (
                    self.model.tokenizer()
                    .detokenize(self.new_tokens + [t], special=True)
                    .decode("utf-8")
                )
                if self.new_str == new_string and not t == self.eos_token_id():
                    continue

                resp = test_against_grammar(new_string, self.lark_grammar)
                if resp[0]:
                    filtered_logits[t] = log_p
            except UnicodeDecodeError as e:
                # print(f"Unicode decode error for token {t}: {e}")
                pass
        return filtered_logits

    def create_child(self, token: int, log_p: float):
        child = BeamSearchNode(
            self.model,
            self.grammar,
            self.cfg,
            lark_grammar=self.lark_grammar,
            depth=self.depth + 1,
            input_tokens=self.input_tokens,
        )
        child.log_p = log_p
        child.token = token
        child.token_str = (
            self.model.tokenizer().detokenize([token], special=True).decode("utf-8")
        )
        next_tokens = self.input_tokens + self.new_tokens + [token]
        child.new_tokens = next_tokens[len(self.input_tokens) :]
        child.new_str = (
            self.model.tokenizer()
            .detokenize(child.new_tokens, special=True)
            .decode("utf-8")
        )
        return child

    def explore_tree(self, tokens: list[int], depth=0):
        tokens = tokens.copy()
        if self.input_tokens is None:
            self.input_tokens = tokens.copy()
        oversample = (
            self.cfg.oversample
            if self.cfg.oversample > 0
            else self.model.metadata.get("tokenizer.vocab_size", 0)
        )
        prompt_str = (
            self.model.tokenizer().detokenize(tokens, special=True).decode("utf-8")
        )
        # print(
        #     "Creating completion with string:",
        #     prompt_str,
        #     f"at depth {depth} with oversample {oversample} from {tokens=}",
        # )
        response = next(
            self.model._create_completion(
                prompt_str,
                max_tokens=1,
                logprobs=oversample,
            ),
            None,
        )
        if response is None:
            # set remaining probs to 1 (?)
            return
        logits: list[dict[int, float]] = response["choices"][0]["logprobs"][
            "top_logprob_tokens"
        ]
        if oversample <= 0:
            oversample = len(logits)
        logits = logits[:oversample] if len(logits) > oversample else logits
        filtered_logits: dict[int, float] = {}

        if response["choices"][0]["finish_reason"] == "stop":
            print(
                f"Finish reason stop reached at {depth=} with {tokens=} and {self.new_str=}, generated response: {response}"
            )
            self.children = [self.create_child(self.eos_token_id(), 0)]
            return
        if len(logits) == 0:
            print(
                f"No logits returned at {depth=} and {self.new_str=} , response: {response['choices'][0]['text']} probs {response['choices'][0]['logprobs']['top_logprob_tokens']}, {oversample=} with {tokens=} "
            )
        filtered_logits = self.filter_logits(logits[0])

        if len(filtered_logits) == 0:
            print(
                f"No valid tokens found at reached at {depth=}  on str {self.new_str=} resp txt {response['choices'][0]['text']}, setting to filtered_logits to {filtered_logits=} (from logits {logits=}) with {tokens=} and oversample {oversample=}"
            )
            resp_detokenized = self.model.tokenizer().tokenize(
                response["choices"][0]["text"].encode("utf-8"),
                special=True,
                add_bos=False,
            )
            reps_tokens = {t: -100 for t in resp_detokenized}
            filtered_logits = self.filter_logits(reps_tokens)

        skip_depth = False
        if self.cfg.k_progress is not None:
            top_k = (
                self.cfg.k_progress[depth]
                if depth < len(self.cfg.k_progress)
                else self.cfg.k_progress[-1]
            )
            skip_depth = top_k == -1
        else:
            top_k = self.cfg.top_k
        if top_k < 0:
            top_k = 1
        top_k_tokens = sorted(
            filtered_logits.items(), key=lambda x: x[1], reverse=True
        )[:top_k]
        for t, log_p in top_k_tokens:
            child = self.create_child(t, log_p)
            assert len(child.new_tokens) == depth + 1
            self.children.append(child)
            if child.is_end():
                print(
                    f"Reached end token {self.cfg.end_token=} or EOS in {child.new_str=}"
                )
                continue
            if depth <= self.cfg.max_depth or top_k <= 0 or skip_depth:
                if depth > 20:
                    print(f"Depth {depth} reached with {child.new_str=}")
                child.explore_tree(child.input_tokens + child.new_tokens, depth + 1)
            else:
                print(f"Max depth reached at {depth=}, {child.new_str=}")

    def best_tree(self, acc_log_p=0):
        if len(self.children) == 0:
            return acc_log_p, self
        best_p = -inf
        best_nd = self
        for c in self.children:
            c_log_p = acc_log_p + c.log_p
            c_p, c_c = c.best_tree(acc_log_p=c_log_p)

            # if len(c_c.new_tokens) == 0:
            #     continue
            if self.cfg.k_progress is not None and self.cfg.k_progress[-1] == -1:
                # require that the last token is the end token or that the string ends with the end token
                if not (c_c.is_end()):
                    continue

            if c_p > best_p:
                best_p = c_p
                best_nd = c_c
        return best_p, best_nd
