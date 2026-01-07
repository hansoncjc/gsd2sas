import sys
import unittest
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(repo_root))

    suite = unittest.defaultTestLoader.discover("tests", pattern="structurefactor_test.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if result.wasSuccessful():
        print("TESTS: PASS")
        return 0
    print("TESTS: FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
