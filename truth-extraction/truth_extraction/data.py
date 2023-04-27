import random
from dataclasses import dataclass

from datasets import ClassLabel, load_dataset
from torch.utils.data import DataLoader, Dataset, Subset

random.seed(0)


@dataclass
class NaturalLanguagePrompt:
    question: str
    answer: str


def prompt_function_imdb(
    text: str, label: str, possible_labels: list[str] = ["pos", "neg"]
) -> NaturalLanguagePrompt:
    return NaturalLanguagePrompt(
        question=f"Is the sentiment for the following review {' or '.join(possible_labels)}?:\n{text}",
        answer=label,
    )


@dataclass
class ContrastSample:
    positive_prompt: NaturalLanguagePrompt
    negative_prompt: NaturalLanguagePrompt
    true_label: str


class ContrastDataset(Dataset[ContrastSample]):
    def __init__(
        self,
        dataset_path: str = "imdb",
        dataset_split: str = "test",
        prompt_function=prompt_function_imdb,
    ) -> None:
        super().__init__()

        # Load raw dataset
        self.raw_dataset: Dataset = load_dataset(dataset_path)[dataset_split]
        assert "text" in self.raw_dataset.features
        assert "label" in self.raw_dataset.features

        assert isinstance(self.raw_dataset.features["label"], ClassLabel)
        assert (
            len(self.raw_dataset.features["label"].names) == 2
        ), "There should only be two possible answers: truth-y or false-y"
        self.label_mapping: list[str] = self.raw_dataset.features["label"].names

        self.prompt_function = prompt_function

    def __getitem__(self, index: int):
        """Map raw_dataset samples to a contrast pair."""
        sample = self.raw_dataset[index]
        text, label_int = sample["text"], sample["label"]
        # Convert to label instead of integer for clarity
        true_label = self.label_mapping[label_int]

        negative_sample = {"text": text, "label": self.label_mapping[0]}
        positive_sample = {"text": text, "label": self.label_mapping[1]}

        # Create a contrast pair: prompt to ask if a sample is correct/truthful
        negative_prompt = self.prompt_function(**negative_sample)
        positive_prompt = self.prompt_function(**positive_sample)

        return (
            positive_prompt.question,
            positive_prompt.answer,
            negative_prompt.question,
            negative_prompt.answer,
            true_label,
        )

    def __len__(self):
        return len(self.raw_dataset)


def get_dataloader(
    tokenizer,
    num_examples: int = 1_000,
    batch_size: int = 16,
    pin_memory: bool = True,
    num_workers: int = 1,
) -> DataLoader:
    """Randomly choose a subset of ContrastDataset that fits some constraints."""
    dataset = ContrastDataset()

    # Keep looking for valid indices until we have `num_examples`
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    valid_indices: list[int] = []
    for i in indices:
        # Tokenized prompt + response should be samller than `model_max_length` tokens
        sample = dataset[i]

        # TODO: DataLoader needs to return tensors, numpy arrays, numbers, dicts, or
        # lists. Why? And how should I make this more natural?
        for question, answer in ((sample[0], sample[1]), (sample[2], sample[3])):
            input_ = question + " " + answer
            if len(tokenizer.encode(input_, truncation=False)) > tokenizer.model_max_length - 2:
                continue

        valid_indices.append(i)
        if len(valid_indices) < num_examples:
            break

    subset_dataset = Subset(dataset, valid_indices)

    return DataLoader(
        subset_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )
