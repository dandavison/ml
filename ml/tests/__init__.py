import unittest

class TestCase(unittest.TestCase):

    def assertEqualArrays(self, A, B, msg=None):
        self.assertTrue(A.shape == B.shape, msg)
        self.assertTrue((A == B).all(), msg)
