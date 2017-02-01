#!/usr/bin/env python

import unittest
from palindrome.tools import is_palindrome

class Tests(unittest.TestCase):
     def test_negative(self):
         self.assertFalse(is_palindrome(1234))

     def test_positive(self):
         self.assertTrue(is_palindrome(1234321))

     def test_single_digit(self):
         for i in range(10):
             self.assertTrue(is_palindrome(i))
   
#if __name__ == '__main__':
#    unittest.main()
