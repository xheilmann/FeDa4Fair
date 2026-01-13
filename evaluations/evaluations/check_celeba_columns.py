from datasets import load_dataset

ds = load_dataset("flwrlabs/celeba", split="train", streaming=True)
print(next(iter(ds)).keys())
