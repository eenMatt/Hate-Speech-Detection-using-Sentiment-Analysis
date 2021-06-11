import os
import random

from transformers import Adafactor, T5ForConditionalGeneration, T5Tokenizer
from main import generate
import torch
import math
import statistics

if __name__ == "__main__":
    directory = os.getcwd()

    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model = T5ForConditionalGeneration.from_pretrained(
        directory,
        return_dict=True,
    )
    # TODO: Make this a method
    if torch.cuda.is_available():
        dev = torch.device("cuda:0")
        print("Running on the GPU")
    else:
        dev = torch.device("cpu")
        print("Running on the CPU")

    model.to(dev)


    results = []
    total = 0
    successes = 0

    with open(os.path.join(directory, "Datasets", "ethos_data", "Ethos_Dataset_Binary.csv"), encoding='cp850') as file:
        lines = file.readlines()[1:]

    length = len(lines)
    result_dict = {}
    # for item in random.sample(files["test"], k=10):
    items = 0
    for index, item in enumerate(lines):
        separated = item.split(";")
        if item[0] == '"':
            separated = [";".join(separated[0:-2]), separated[-1]]
        tweet = separated[0].strip()
        label = separated[1].strip()
        label = float(label)
        label = int(label >= 0.5)
        label = ["NOT", "OFF"][label]
        
        output = generate(tweet, model, tokenizer, dev)
        output = output[6:-4]
        success = label == output.split()[0]
        # print(as_testing(item), item["answer"], output)
        result = (label, output, success)
        print(result)
        level = round(float(separated[1].strip()), 1)
        if level < 0.5:
            level = 1 - level
        if level in result_dict:
            result_dict[level].append(success)
        else:
            result_dict[level] = [success]
        results.append(str(result) + "\n")
        total += 1
        successes += success
        if index % 100 == 1:
            print(
                f"{round(index/length*100)}% Done; {round(successes/total*100)}% accuracy")

    with open("results.txt", "w") as file:
        file.writelines(results)
    print(
        f"FINAL RESULTS: {successes} correct out of {total}, for a success rate of {round(successes/total*100)}%")
    print(result_dict)
    result_dict = {key: statistics.mean(value) for key, value in result_dict.items()}
    print(result_dict)
