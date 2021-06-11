import json
import os
import random

import pandas as pd
import torch
from transformers import Adafactor, T5ForConditionalGeneration, T5Tokenizer


def as_question(question, context):
    nl = "\n"
    return f'{nl.join(context.split())}\n{question["text"][:-1]}?'


def read_folder(folder_name):
    sets = ["dev", "train", "test"]
    filenames = [os.path.join(folder_name, set_name + ".jsonl")
                 for set_name in sets]
    contents = []
    dicts = {}
    for i in range(len(filenames)):
        filename = filenames[i]
        with open(filename) as file:
            dicts[sets[i]] = [json.loads(line) for line in file.readlines()]
    return dicts


def generate(text, model, tokenizer, device):
    model.eval()
    input_ids = tokenizer.encode(text, return_tensors="pt")
    input_ids = input_ids.to(device)
    outputs = model.generate(input_ids)
    return tokenizer.decode(outputs[0])


if __name__ == "__main__":
    random.seed(43)
    sets = ["dev", "train", "test"]
    directory = os.getcwd()
    # files = read_folder(os.path.join(directory,"nabduction", "depth-5"))
    # files = read_folder(os.path.join(directory, "Abduction-Animal"))
    inputs = []
    labels = []
    nl = "\n"
    with open(os.path.join(directory, "Datasets", "OLIDv1.0", "olid-training-v1.0.tsv"), encoding='cp850') as file:
        train_lines = file.readlines()[1:]
    # train_lines = random.sample(train_lines, k=10) # TODO: TURN OFF
    for line in train_lines:
        items = line.strip().split("\t")
        inputs.append(items[1])
        labels.append(" ".join(items[2:]))
    batch_size = 5
    num_of_batches = len(inputs) // batch_size

    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model = T5ForConditionalGeneration.from_pretrained(
        "t5-base", return_dict=True)
    optimizer = Adafactor(
        model.parameters(),
        lr=1e-3,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.0,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
    )


    if torch.cuda.is_available():
        dev = torch.device("cuda:0")
        print("Running on the GPU")
    else:
        dev = torch.device("cpu")
        print("Running on the CPU")

    model.to(dev)
    model.train()

    loss_per_10_steps = []
    num_of_epochs = 10
    for epoch in range(1, num_of_epochs + 1):
        print("Running epoch: {}".format(epoch))

        running_loss = 0

        for i in range(num_of_batches):
            inputbatch = inputs[i * batch_size: i * batch_size + batch_size]
            labelbatch = labels[i * batch_size: i * batch_size + batch_size]
            inputbatch = tokenizer.batch_encode_plus(
                inputbatch, padding=True, max_length=400, return_tensors="pt"
            )["input_ids"]
            labelbatch = tokenizer.batch_encode_plus(
                labelbatch, padding=True, max_length=400, return_tensors="pt"
            )["input_ids"]
            inputbatch = inputbatch.to(dev)
            labelbatch = labelbatch.to(dev)

            # clear out the gradients of all Variables
            optimizer.zero_grad()

            # Forward propogation
            outputs = model(input_ids=inputbatch, labels=labelbatch)
            loss = outputs.loss
            loss_num = loss.item()
            #logits = outputs.logits
            running_loss += loss_num
            # if i % 10 == 0:
            #     loss_per_10_steps.append(loss_num)
            # out.update(progress(loss_num, i, num_of_batches + 1))
            if i % 100 == 0:
                print(
                    f"{i} batches done of {num_of_batches}; epoch {epoch} of {num_of_epochs}")

            # calculating the gradients
            loss.backward()

            # updating the params
            optimizer.step()

        running_loss = running_loss / int(num_of_batches)
        print("Epoch: {} , Running loss: {}".format(epoch, running_loss))

    print("Training complete, saving model")
    torch.save(model.state_dict(), "pytorch_model.bin")

    print("Beginning testing")

    results = []
    total = 0
    successes = 0

    for number in range(3):
        letter = "abc"[number]
        with open(os.path.join(directory, "Datasets", "OLIDv1.0", f"testset-level{letter}.tsv"), encoding='cp850') as file:
            tweets = file.readlines()[1:]
        with open(os.path.join(directory, "Datasets", "OLIDv1.0", f"labels-level{letter}.csv"), encoding='cp850') as file:
            labels = file.readlines()

        tests = []
        for i in range(len(tweets[1:])):
            tweet_info = tweets[i].split("\t")
            label_info = labels[i].split(",")
            if tweet_info[0] != label_info[0]:
                raise ValueError
            tests.append((tweet_info[1].strip(), label_info[1].strip()))

        

        length = len(tests)
        items = 0
        # for index, (tweet, label) in enumerate(random.sample(tests, k=10)):# TODO: TURN OFF
        for index, (tweet, label) in enumerate(tests):
            output = generate(tweet, model, tokenizer, dev)
            # print(output)# TODO: TURN OFF
            output = output[6:-4]
            try:
                success = label == output.split()[number] # Only tests one part of output at a time
            except IndexError:
                success = False
            # print(as_testing(item), item["answer"], output)
            result = (label, output, success)
            results.append(str(result) + "\n")
            total += 1
            successes += success
            if index % 100 == 1:
                print(
                    f"{round(index/length*100)}% Done of set {letter}; {round(successes/total*100)}% accuracy")

    print("Testing finished, writing results to file")
    with open("results.txt", "w") as file:
        file.writelines(results)
    print(
        f"FINAL RESULTS: {successes} correct out of {total}, for a success rate of {round(successes/total*100)}%")
