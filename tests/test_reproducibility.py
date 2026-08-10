import os
import tempfile
import unittest

import numpy as np

import cpnest
import cpnest.model


class GaussianModel(cpnest.model.Model):
    names = ['x']
    bounds = [[-5, 5]]

    def log_likelihood(self, p):
        return -0.5*p['x']**2


class ReproducibilityTestCase(unittest.TestCase):
    def run_sampler(self, output, nthreads):
        work = cpnest.CPNest(
            GaussianModel(),
            output   = output,
            verbose  = 0,
            seed     = 1234,
            nlive    = 17,
            nthreads = nthreads,
            poolsize = 16,
            maxmcmc  = 16,
            dlogZ    = 0.5,
        )
        try:
            work.run()
            terminal_state = (
                work.NS.iteration,
                work.NS.state.iteration,
                work.NS.logZ,
                work.NS.state.info,
            )
            return (
                work.nested_samples.copy(),
                np.asarray(work.NS.insertion_indices),
                terminal_state,
            )
        finally:
            work.manager.shutdown()

    def test_fixed_seed_is_exactly_reproducible(self):
        for nthreads in (1, 4):
            with self.subTest(nthreads=nthreads), tempfile.TemporaryDirectory() as output:
                first = self.run_sampler(os.path.join(output, 'first'), nthreads)
                second = self.run_sampler(os.path.join(output, 'second'), nthreads)

                np.testing.assert_array_equal(first[0], second[0])
                np.testing.assert_array_equal(first[1], second[1])
                self.assertEqual(first[2], second[2])


if __name__ == '__main__':
    unittest.main(verbosity=2)
