"""inscribed.certify is the exact verifier; it must import without the solver stack."""

import subprocess
import sys
import unittest


class TestCertifyImports(unittest.TestCase):
    def test_certify_does_not_import_cvxpy(self):
        """Checked in a fresh interpreter, so no test-order import can mask it."""
        script = (
            "import sys; import inscribed.certify; "
            "assert 'cvxpy' not in sys.modules, 'inscribed.certify pulled in cvxpy'"
        )
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
