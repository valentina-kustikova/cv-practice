import torchvision
import random
import json

rng = random.Random(42)
ds = torchvision.datasets.Caltech101(root='data', download=False)

classes = [[] for _ in range(101)]
for idx, label in enumerate(ds.y):
    classes[label].append(idx)

train_idx, test_idx = [], []

for i in classes:
    rng.shuffle(i)
    train_idx.extend(i[:20])
    test_idx.extend(i[20:])

with open('split.json', 'w') as f:
    json.dump({'train': train_idx, 'test': test_idx}, f)








