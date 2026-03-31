from __future__ import annotations

import unittest

from stress_harness.runtime import ContainerRuntimeInspector


class ContainerRuntimeInspectorTests(unittest.TestCase):
    def test_api_host_port(self) -> None:
        inspector = ContainerRuntimeInspector("http://127.0.0.1:8080/v1/chat/completions")
        self.assertEqual(inspector.api_host_port(), 8080)

    def test_host_port_matches(self) -> None:
        ports = "0.0.0.0:8080->8080/tcp, [::]:8080->8080/tcp"
        self.assertTrue(ContainerRuntimeInspector.host_port_matches(ports, 8080))
        self.assertFalse(ContainerRuntimeInspector.host_port_matches(ports, 9090))


if __name__ == "__main__":
    unittest.main()
