import unittest

from models.utils.lists import split_list, sort_by


class ListsTestCase(unittest.TestCase):
    def test_SplitLists(self):
        self.assertEqual([[1, 2, 3]], split_list([1, 2, 3], 1))
        self.assertEqual([[1, 2, 3], [4, 5, 6]], split_list([1, 2, 3, 4, 5, 6], 2))
        self.assertEqual([[1, 2, 3], [4, 5]], split_list([1, 2, 3, 4, 5], 2))
        self.assertEqual([[1, 2, 3], [4, 5, 6], [7, 8, 9]], split_list([1, 2, 3, 4, 5, 6, 7, 8, 9], 3))
        self.assertRaises(ValueError, lambda: split_list([1, 2, 3], 5))

    def test_SortBy(self):
        self.assertEqual([1], sort_by([2], [1]))
        self.assertEqual([3, 4], sort_by([2, 1], [4, 3]))
        self.assertEqual([6, 4, 5], sort_by([2, 3, 1], [4, 5, 6]))
        self.assertEqual([5, 4, 6], sort_by([2, 3, 1], [4, 5, 6], reverse=True))
