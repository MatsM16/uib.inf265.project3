import torch
from torch.utils.data import DataLoader, TensorDataset

def dataset_create(
    words: list[str],
    context_size_before: int = 0,
    context_size_after: int = 0,
    vocabulary_index_to_target: dict[int, int] = {},
    dataset_name: str | None = None
) -> TensorDataset:
    """
    Creates a dataset from the given words.
    """

    filename = DATA_DIR + f'dataset/{dataset_name}.pt'
    if os.path.exists(filename) and dataset_name is not None and not FORCE_RETRAIN:
        return torch.load(filename)

    word_idx = [vocabulary[word] for word in words]

    contexts = []
    targets = []
    for i in range(context_size_before, len(words) - context_size_after - 1):

        context_before = word_idx[i-context_size_before:i]
        context_after = word_idx[i+1:i+1+context_size_after]
        context = context_before + context_after
        target = word_idx[i+context_size_before]
        target = vocabulary_index_to_target.get(target, None)

        if target is not None:
            contexts.append(torch.tensor(context))
            targets.append(target)

    contexts = torch.stack(contexts).to(device)
    targets = torch.tensor(targets).to(device)

    dataset = TensorDataset(contexts, targets)
    torch.save(dataset, filename)

    return dataset