import sentencepiece as spm
import torch

from ...tokenizers import BaseTokenizer


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
        )
        print(f"Finished training LLaMa Tokenizer")

    def get_vocab(self) -> dict[str, int]:
        return self.vocab

    def get_vocab_size(self) -> int:
        return len(self.vocab)

    def _tokenize(self, text: str) -> list[str]:
        return self.tokenizer.encode_as_pieces(
            text, add_bos=self.add_bos_token, add_eos=self.add_eos_token
        )

    def _convert_token_to_id(self, token: str) -> int:
        return self.tokenizer.piece_to_id(token)

    def _convert_id_to_token(self, token_id: int) -> str:
        return self.tokenizer.id_to_piece(token_id)

    def _convert_tokens_to_string(self, tokens: list[str]) -> str:
        return self.tokenizer.decode(tokens)

    def build_inputs_with_special_tokens(self, token_ids: list[int]) -> list[int]:
        return token_ids

    def num_special_tokens_to_add(self):
        return 0

    def _encode(self, text: str, **kwargs) -> list[int]:
        return self.tokenizer.encode_as_ids(
            text, add_bos=self.add_bos_token, add_eos=self.add_eos_token
        )

    def _decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids)

    def _batch_decode(self, sequences: list[list[int]]) -> list[str]:
        return self.tokenizer.decode(sequences)
