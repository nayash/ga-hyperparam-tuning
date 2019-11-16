#
# Copyright (c) 2019. Asutosh Nayak (nayak.asutosh@ymail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#

from abc import ABC, abstractmethod, ABCMeta


class GAAbstract(ABC):
    __metaclass__ = ABCMeta

    @property
    @abstractmethod
    def target(self):
        raise NotImplementedError

    @target.setter
    @abstractmethod
    def target(self, target):
        pass

    @property
    @abstractmethod
    def population(self):
        raise NotImplementedError

    @target.setter
    @abstractmethod
    def population(self, pop):
        pass

    def selection(self):
        pass

    def mutation(self, individual):
        pass

    def cross_over(self, individual1, individual2):
        pass

    def run(self):
        pass

    def should_exit(self):
        pass

    def calc_fitness_score(self, individual1, individual2):
        pass

