import random


with open("entertainment_articles.txt", "r") as f:
    text = f.read()

articles = text.split("\n\n")
random.seed(2424)
random.shuffle(articles)

training_size = int(len(articles)*.8)
val_size = int(len(articles)*.1)
test_size = int(len(articles)*.1)

training = articles[:training_size]
val = articles[training_size:val_size+training_size]
test = articles[val_size+training_size:]

with open('train.txt', 'w') as f:
    for line in training:
        f.write(f"{line}\n")
with open('test.txt', 'w') as f:
    for line in test:
        f.write(f"{line}\n")
with open('val.txt', 'w') as f:
    for line in val:
        f.write(f"{line}\n")