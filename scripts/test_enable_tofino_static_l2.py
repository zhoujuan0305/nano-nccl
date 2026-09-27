import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).with_name("enable_tofino_static_l2.py")
SPEC = importlib.util.spec_from_file_location("enable_tofino_static_l2", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class StaticL2Tests(unittest.TestCase):
    def test_builds_only_the_two_reciprocal_entries(self):
        desired = MODULE.build_desired_entries(
            11, "02:00:00:00:00:0a", 22, "02:00:00:00:00:0b"
        )
        self.assertEqual(
            desired,
            {
                (11, int("02000000000b", 16)): 22,
                (22, int("02000000000a", 16)): 11,
            },
        )

    def test_rejects_same_port_or_same_mac(self):
        with self.assertRaises(ValueError):
            MODULE.build_desired_entries(11, "02:00:00:00:00:0a", 11, "02:00:00:00:00:0b")
        with self.assertRaises(ValueError):
            MODULE.build_desired_entries(11, "02:00:00:00:00:0a", 22, "02:00:00:00:00:0a")

    def test_mac_parser_rejects_multicast(self):
        with self.assertRaises(Exception):
            MODULE.parse_mac("01:00:00:00:00:01")

    def test_builds_proxy_arp_for_both_same_subnet_endpoints(self):
        desired = MODULE.build_desired_arp(
            "192.0.2.10", "02:00:00:00:00:0a",
            "192.0.2.11", "02:00:00:00:00:0b",
        )
        self.assertEqual(
            desired,
            {
                (1, 0xC000020A): 0x02000000000A,
                (1, 0xC000020B): 0x02000000000B,
            },
        )

    def test_rejects_different_subnets(self):
        with self.assertRaises(ValueError):
            MODULE.build_desired_arp(
                "192.0.2.10", "02:00:00:00:00:0a",
                "192.0.3.11", "02:00:00:00:00:0b",
            )

    def test_dry_run_never_connects(self):
        old_connect = MODULE._connect
        try:
            MODULE._connect = lambda _args: self.fail("dry-run must not connect to BFRT")
            result = MODULE.main([
                "--dev-port-a", "11",
                "--ip-a", "192.0.2.10",
                "--mac-a", "02:00:00:00:00:0a",
                "--dev-port-b", "22",
                "--ip-b", "192.0.2.11",
                "--mac-b", "02:00:00:00:00:0b",
            ])
            self.assertEqual(result, 0)
        finally:
            MODULE._connect = old_connect


if __name__ == "__main__":
    unittest.main()
