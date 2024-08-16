import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import json


class ArcDataset(Dataset):
    def __init__(self):
        # A "grid" is a rectangular matrix (list of lists) of integers between 0 and 9 (inclusive).
        # The smallest possible grid size is 1x1 and the largest is 30x30.
        self.task_examples = []  # combination of combinations of example tasks and their solutions
        self.actual_tasks = []  # combination of actual tasks
        self.actual_solutions = []  # combination of the actual solutions, located in arc-agi_training_solutions.json

        with open("data/arc-agi_training_challenges.json") as f:
            data = json.load(f)
            for idx in data:
                #print(data[idx])
                self.task_examples.append(data[idx]["train"])
                self.actual_tasks.append(data[idx]["test"][0]["input"])

        with open("data/arc-agi_training_solutions.json") as f:
            data = json.load(f)
            for idx in data:
                self.actual_solutions.append(data[idx][0])

    def __len__(self):
        return len(self.actual_tasks)

    def __getitem__(self, idx):
        examples = self.task_examples[idx]
        task = self.actual_tasks[idx]
        solution = self.actual_solutions[idx]

        #print("examples\n", examples)
        #print("task\n", task)
        # print("solution\n", solution)

        output_example = []
        for example in examples:
            output_example.append(example["output"])

        # Convert the matrices to torch tensors
        print("output_example\n", output_example)
        print("actual_task\n", task)
        print("solution\n", solution)

        #np.array(output_example)
        #np.array(task)
        #np.array(solution)

        output_example = torch.tensor(output_example, dtype=torch.float32)
        task = torch.tensor(task, dtype=torch.float32)
        solution = torch.tensor(solution, dtype=torch.float32)
        print("output_example tensor\n", output_example)
        print("actual_task tensor\n", task)
        print("solution tensor\n", solution)

        # exit()
        return output_example, task, solution


dataset = ArcDataset()
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

for data in dataloader:
    pass  # triggers getitem for validation of the tensors
