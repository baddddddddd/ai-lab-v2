import sentencepiece as spm
import torch

from ...tokenizers import BaseTokenizer

SPIECE_UNDERLINE = "▁"


class LlamaTokenizer(BaseTokenizer):
    def __init__(
        self,
        vocab_file: str,
        unk_token: str = "<unk>",
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        add_bos_token: bool = True,
        add_eos_token: bool = False,
    ):

        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token

        self.tokenizer = spm.SentencePieceProcessor(model_file=vocab_file)

        self.vocab = {
            self.tokenizer.id_to_piece(id_): id_
            for id_ in range(self.tokenizer.get_piece_size())
        }

        self.special_tokens = set(
            [tok for tok in [unk_token, bos_token, eos_token] if tok is not None]
        )
        super().__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
        )

    @staticmethod
    def train(
        corpus_file: str,
        model_prefix: str = "tokenizer",
        vocab_size: str = 32000,
        character_coverage: float = 0.99995,
        add_dummy_prefix: bool = False,
        unk_token: str = "<unk>",
        bos_token: str = "<s>",
        eos_token: str = "</s>",
    ):
        import multiprocessing

        print("Training LLaMa Tokenizer...")
        spm.SentencePieceTrainer.train(
            input=corpus_file,
            model_prefix=model_prefix,
            model_type="bpe",
            vocab_size=vocab_size,
            unk_piece=unk_token,
            bos_piece=bos_token,
            eos_piece=eos_token,
            self_test_sample_size=0,
            input_format="text",
            character_coverage=character_coverage,
            num_threads=multiprocessing.cpu_count(),
            split_digits=True,
            allow_whitespace_only_pieces=True,
            byte_fallback=True,
            unk_surface=" \uFFF7 ",
            normalization_rule_name="identity",
            add_dummy_prefix=add_dummy_prefix,
        )
        print(f"Finished training LLaMa Tokenizer")

    def get_vocab(self) -> dict[str, int]:
        return self.vocab

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _tokenize(self, text: str) -> list[str]:
        return self.tokenizer.encode_as_pieces(text)

    def _convert_token_to_id(self, token: str) -> int:
        return self.tokenizer.piece_to_id(token)

    def _convert_id_to_token(self, token_id: int) -> str:
        return self.tokenizer.id_to_piece(token_id)

    def _convert_tokens_to_string(self, tokens: list[str]) -> str:
        out = ""

        sub_tokens = []
        for tok in tokens:
            if tok in self.special_tokens:
                out += self.tokenizer.decode(sub_tokens) + tok
                sub_tokens.clear()
            else:
                if not sub_tokens and tok.startswith(SPIECE_UNDERLINE):
                    out += " "

                sub_tokens.append(tok)

        out += self.tokenizer.decode(sub_tokens)
        return out

    def build_inputs_with_special_tokens(self, token_ids: list[int]) -> list[int]:
        return token_ids

    def num_special_tokens_to_add(self):
        return 0

    def _encode(self, text: str, **kwargs) -> list[int]:
        return self.tokenizer.encode_as_ids(
            text, add_bos=self.add_bos_token, add_eos=self.add_eos_token
        )

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]] | list[list[dict[str, str]]],
        tokenize: bool = True,
        add_generation_prompt: bool = False,
    ):
        user_label = "Human: "
        assistant_label = "\n\nAI: "
        tokenized_assistant_label = self._tokenize(assistant_label)

        if isinstance(conversation, list) and isinstance(conversation[0], dict):
            is_batched = False
            conversation_list = [conversation]
        else:
            is_batched = True
            conversation_list = conversation

        batched_tokens = []
        for conversation in conversation_list:
            is_user_turn = True
            conversation_tokens = []
            for turn in conversation:
                role = turn["role"]
                if (is_user_turn and role != "user") or (
                    not is_user_turn and role != "assistant"
                ):
                    raise ValueError(
                        "Conversation roles must alternate user/assistant/user/assistant/..."
                    )

                content = turn["content"]

                if is_user_turn:
                    prepend, append = [self.bos_token], []
                    template = user_label + content
                else:
                    prepend, append = [], [self.eos_token]
                    template = assistant_label + content

                tokens = prepend + self._tokenize(template) + append
                conversation_tokens += tokens

                is_user_turn = not is_user_turn

            if add_generation_prompt:
                conversation_tokens += tokenized_assistant_label

            batched_tokens.append(conversation_tokens)

        if tokenize:
            batched_result = [
                self.convert_tokens_to_ids(tokens) for tokens in batched_tokens
            ]
        else:
            batched_result = [
                self._convert_tokens_to_string(tokens) for tokens in batched_tokens
            ]

        return batched_result if is_batched else batched_result[0]
