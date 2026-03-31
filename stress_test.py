#!/usr/bin/env python3

from stress_harness import ConsoleReporter, StressConfig, StressTestRunner


def main() -> int:
    config = StressConfig.from_env()
    reporter = ConsoleReporter()
    result = StressTestRunner(config, reporter).run()
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
