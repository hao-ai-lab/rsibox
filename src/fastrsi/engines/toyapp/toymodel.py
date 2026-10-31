"""The code that edits target."""


class Accumulator:
    def __init__(self):
        self.total = 0

    def add(self, value):
        self.total += value
        return self.total


def forward(x):
    return x * 2 + 1


def postprocess(values):
    return [value + 0 for value in values]


def unused(value):
    return value * 7
