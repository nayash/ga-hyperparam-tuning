#
# Copyright (c) 2019. Asutosh Nayak (nayak.asutosh@ymail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#

from ga_engine import GAEngine
import numpy as np


mutation_only_iterations = []
full_ga_iterations = []

for i in range(1000):
    mutation_only_iterations.append(GAEngine(10).run(True))
    full_ga_iterations.append(GAEngine(10).run())

print("Average generations taken to reach target: mutations_only = {} and full_ga = {}".format(
    np.mean(mutation_only_iterations), np.mean(full_ga_iterations)))
