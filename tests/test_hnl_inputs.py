"""Majorana charge accounting and mass-consistent HNL matrix elements."""
import unittest
import numpy as np
from exhad.full_decays import hnl_tables
from exhad.core.kinematics import matrix_element


class HNLInputs(unittest.TestCase):
    def test_two_pion_width_counts_each_charge_once(self):
        rows, widths = hnl_tables()
        gamma = np.interp(1., widths[:, 0], widths[:, 1])
        pair = [next(r for r in rows if r[0] == name)
                for name in ('2Pie', '2Piebar')]
        partials = [gamma*np.interp(1., r[2][:,0], r[2][:,1]) for r in pair]
        # Independent evaluation of the source's rho-pole formula at 1 GeV.
        np.testing.assert_allclose(partials, [7.441409825480385e-14]*2, rtol=2e-14)
        self.assertAlmostEqual(sum(partials)/7.441409825480385e-14, 2.)

    def test_strange_current_uses_the_supplied_daughter_masses(self):
        rows, _ = hnl_tables()
        expression = next(r[5] for r in rows if r[0] == 'Jets-use')
        rng = np.random.default_rng(812)
        for masses in ((.0022,.104,.000511),(.0022,.095,.000511)):
            m1,m2,m3 = masses
            f = matrix_element(expression, masses)
            for mass in (1.38,2.,5.):
                e1 = m1 + rng.random(4096)*((mass**2+m1*m1-(m2+m3)**2)/(2*mass)-m1)
                s = mass**2+m1*m1-2*mass*e1
                c = (mass-e1)*(s+m3*m3-m2*m2)/(2*s)
                h = np.sqrt((e1*e1-m1*m1)*(s-(m2+m3)**2)*(s-(m2-m3)**2))/(2*s)
                values = f(mass,e1,c+h*(2*rng.random(len(e1))-1))
                self.assertTrue(np.all(np.isfinite(values) & (values>=0)))


if __name__ == '__main__':
    unittest.main()
