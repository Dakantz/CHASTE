import logging

from llama_cpp import (
    ChatCompletionRequestMessage,
    Llama,
    LlamaGrammar,
    llama_chat_format,
)
from tqdm import tqdm

from llama_cpp_beamsearch.beam_search import BeamSearchNode
from llama_cpp_beamsearch.config import BeamSearchConfig
from llama_cpp_beamsearch.grammar import GBNF_PARSER, test_against_grammar

logger = logging.getLogger(__name__)


class BeamSearchCompletion:
    def __init__(
        self,
        model: Llama,
        grammar: LlamaGrammar,
        config: BeamSearchConfig | None = None,
        loglevel=logging.INFO,
    ):
        self.config = config if config is not None else BeamSearchConfig()
        self.grammar = grammar
        self.model = model
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(loglevel)
        if model._logits_all is False:
            self.logger.error(
                "Model is not set to return all logits. Beam search may not work as expected."
            )
            raise ValueError(
                "Model is not set to return all logits. Beam search may not work as expected."
            )

    def completion_beam_search_messages(
        self,
        messages: list[ChatCompletionRequestMessage],
        max_tokens=128,
    ):
        input_message = llama_chat_format.format_llama3(messages)
        return self.completion_beam_search(input_message, max_tokens=max_tokens)

    def completion_beam_search(
        self,
        input_message: str,
        max_tokens=128,
    ):
        # get the formatter
        prompt: str = input_message

        input_tokens: list[int] = self.model.tokenizer().tokenize(
            prompt.encode("utf-8"), add_bos=False
        )
        new_tokens = []
        for t in tqdm(range(max_tokens), "Beam-Searching tokens"):
            root = BeamSearchNode(
                self.model,
                grammar=self.grammar,
                cfg=self.config,
                input_tokens=input_tokens,
                new_tokens=new_tokens,
            )
            root.explore_tree(input_tokens + new_tokens)
            t_p, t_nd = root.best_tree()
            old_tokens = new_tokens.copy()
            skip_tokens = None
            if self.config.k_progress is not None and t_p is not None:
                skip_tokens = None
            elif self.config.skip_tokens is not None:
                skip_tokens = self.config.skip_tokens
            else:
                skip_tokens = 1
            target_tokens = None
            if skip_tokens is not None:
                target_tokens = len(old_tokens) + skip_tokens
            new_tokens = t_nd.new_tokens[:target_tokens]

            added_str = self.model.tokenizer().decode(
                t_nd.new_tokens[len(old_tokens) : target_tokens]
            )
            complete_str = self.model.tokenizer().decode(new_tokens)
            if t_nd.eos_token_id() in new_tokens or len(old_tokens) == len(new_tokens):
                break
            print(
                f"Decoded beam search output: {added_str=} {skip_tokens=} {t_nd.new_str=} {complete_str=}"
            )
        decoded = self.model.tokenizer().decode(new_tokens)
        print(f"Final beam search output: {decoded=}")
        return decoded
