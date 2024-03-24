import unittest
import cond_nums

class MyTestCase(unittest.TestCase):
    def test_norm(self):
        mat = [
            [1, 1],
            [1, 1]
        ]

        self.assertEqual(cond_nums.get_norm(mat), 2)  # add assertion here

    def test_get_spectral_number(self):
        mat = [
            [1, 0],
            [0, 1]
        ]
        self.assertAlmostEqual(cond_nums.spectral_number(mat), 2)

    def test_get_volume_number(self):
        mat = [
            [1, 0],
            [0, 1]
        ]
        self.assertAlmostEqual(cond_nums.volume_number(mat), 2)

    def test_get_ang_number(self):
        mat = [
            [1, 0],
            [0, 1]
        ]
        self.assertAlmostEqual(cond_nums.ang_number(mat), 1)

    def test_get_hilbert_mat(self):
        m = cond_nums.get_hilbert_matrix(2)
        self.assertAlmostEqual(m, [[1, 1/2], [1/2, 1/3]])


if __name__ == '__main__':
    unittest.main()
