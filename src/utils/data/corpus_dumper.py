from io import TextIOWrapper
from pathlib import Path

import spacy
from datasets import Dataset as HFDataset, IterableDataset as HFIterableDataset


class CorpusDumper:
    @staticmethod
    def get_spacy_sentencizer(spacy_model: str) -> spacy.Language:
        nlp = spacy.load(spacy_model, disable=["tagger", "parser", "ner", "lemmatizer"])
        nlp.add_pipe("sentencizer")

        return nlp

    @staticmethod
    def write_example(
        f: TextIOWrapper,
        text: str,
        split_by_sentence: bool,
        split_by_newline: bool,
        add_space_prefix: bool,
        separator: str | None,
        nlp: spacy.Language | None,
        **kwargs,
    ) -> int:
        total_written = 0

        if split_by_sentence:
            doc = nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            for sentence in sentences:
                if add_space_prefix:
                    sentence = " " + sentence

                f.write(sentence + "\n")
                total_written += 1
        elif split_by_newline:
            lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
            for line in lines:
                line = line.strip()
                if line:
                    if add_space_prefix:
                        line = " " + line

                    f.write(line + "\n")
                    total_written += 1
        else:
            if add_space_prefix:
                text = " " + text

            f.write(text)
            f.write(separator)
            total_written += 1

        return total_written

    @staticmethod
    def from_huggingface_dataset(
        dataset: HFDataset,
        output_path: str | Path,
        text_column: str = "text",
        separator: str = "\n\n",
        encoding: str = "utf-8",
        chunk_size: int = 1000,
        log_interval: int = 10000,
        add_space_prefix: bool = False,
        split_by_newline: bool = False,
        split_by_sentence: bool = False,
        spacy_model: str = "en_core_web_sm",
    ):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        nlp = None
        if split_by_sentence:
            nlp = CorpusDumper.get_spacy_sentencizer(spacy_model)

        with open(output_path, "w", encoding=encoding) as f:
            total_examples = len(dataset)
            total_written = 0

            for i in range(0, total_examples, chunk_size):
                end_idx = min(i + chunk_size, total_examples)
                chunk = dataset[i:end_idx]

                if isinstance(chunk[text_column], list):
                    texts = chunk[text_column]
                else:
                    texts = [chunk[text_column]]

                for text in texts:
                    total_written += CorpusDumper.write_example(
                        f=f,
                        text=text,
                        split_by_sentence=split_by_sentence,
                        split_by_newline=split_by_newline,
                        add_space_prefix=add_space_prefix,
                        separator=separator,
                        nlp=nlp,
                    )

                if i > 0 and i % log_interval == 0:
                    print(f"Processed {i} examples...")

        print(f"Corpus dumped to {output_path}")
        print(f"Total examples/lines: {total_written}")
        print(f"File size: {output_path.stat().st_size / (1024*1024):.2f} MB")

    @staticmethod
    def from_huggingface_iterable_dataset(
        dataset: HFIterableDataset,
        output_path: str | Path,
        text_column: str = "text",
        max_examples: int | None = None,
        separator: str = "\n\n",
        encoding: str = "utf-8",
        chunk_size: int = 1000,
        log_interval: int = 10000,
        add_space_prefix: bool = False,
        split_by_newline: bool = False,
        split_by_sentence: bool = False,
        spacy_model: str = "en_core_web_sm",
    ):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        nlp = None
        if split_by_sentence:
            nlp = CorpusDumper.get_spacy_sentencizer(spacy_model)

        with open(output_path, "w", encoding=encoding) as f:
            count = 0
            total_written = 0

            for example in dataset:
                if max_examples and count >= max_examples:
                    break

                text = example[text_column]

                total_written += CorpusDumper.write_example(
                    f=f,
                    text=text,
                    split_by_sentence=split_by_sentence,
                    split_by_newline=split_by_newline,
                    add_space_prefix=add_space_prefix,
                    separator=separator,
                    nlp=nlp,
                )

                count += 1
                if count % log_interval == 0:
                    print(f"Processed {count} examples...")

        print(f"Corpus dumped to {output_path}")
        print(f"Total examples/lines: {total_written}")
        print(f"File size: {output_path.stat().st_size / (1024*1024):.2f} MB")
