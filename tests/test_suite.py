import sys

import tests
sys.path.append('../tests')
import unittest
from tests.ga_engine_test import GAEngineTest
from tests.population_test import PopulationTest
from tests.utils_test import UtilsTest


def create_suite():
    return unittest.defaultTestLoader.loadTestsFromModule(tests)
    # test_suite = unittest.TestSuite()
    # test_suite.addTest(UtilsTest())
    # test_suite.addTest(PopulationTest())
    # test_suite.addTest(GAEngineTest())
    # return test_suite


suite = create_suite()
runner = unittest.TextTestRunner()
runner.run(suite)
