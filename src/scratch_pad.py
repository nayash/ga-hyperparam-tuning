import numpy as np


class Test:
    def __init__(self, n):
        self.n = n

    def set_n(self, n):
        self.n = n

    def get_n(self):
        return self.n


def print_ls(ls):
    for i, t in enumerate(ls):
        print(i, t.get_n())


ls = []

for i in range(10):
    ls.append(Test(np.random.randint(0, 50)))

print_ls(ls)
print("____________")
for i, t in enumerate(ls):
    t.set_n(i)

print_ls(ls)

