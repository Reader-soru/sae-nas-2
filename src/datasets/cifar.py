from pathlib import Path

from torchvision import datasets, transforms


def build_cifar_datasets(
    dataset_name: str,
    data_root: str | Path,
    train_transform=None,
    test_transform=None,
):
    """
    Build the torchvision train/test datasets for CIFAR-10 or CIFAR-100.

    The caller supplies transforms explicitly so that preprocessing choices
    remain controlled by the reproduction configuration rather than being
    silently hard-coded here.
    """
    data_root = Path(data_root)

    if train_transform is None:
        train_transform = transforms.ToTensor()

    if test_transform is None:
        test_transform = transforms.ToTensor()

    dataset_name = dataset_name.lower()

    if dataset_name == "cifar10":
        dataset_cls = datasets.CIFAR10
    elif dataset_name == "cifar100":
        dataset_cls = datasets.CIFAR100
    else:
        raise ValueError(
            f"Unsupported CIFAR dataset: {dataset_name}. "
            "Expected 'cifar10' or 'cifar100'."
        )

    train_dataset = dataset_cls(
        root=data_root,
        train=True,
        download=True,
        transform=train_transform,
    )

    test_dataset = dataset_cls(
        root=data_root,
        train=False,
        download=True,
        transform=test_transform,
    )

    return train_dataset, test_dataset