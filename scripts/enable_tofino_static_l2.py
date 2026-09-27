#!/usr/bin/env python3
"""Configure one two-port L2 path on an already running Tofino pipeline.

Dry-run is the default. --configure-ports adds/enables only the two named ports;
existing port settings and forwarding/ARP entries are checked before any write.
"""
from __future__ import print_function

import argparse
import ipaddress
import logging
import os
import re
import sys
import time


DEFAULT_PROGRAM = "l2learn"
DEFAULT_FORWARD_TABLE = "pipe.SwitchIngress.forward"
DEFAULT_ARP_TABLE = "pipe.SwitchIngress.proxy_arp"
INGRESS_PORT_KEY = "ig_intr_md.ingress_port"
DEST_MAC_KEY = "hdr.ethernet.dst_addr"
MAC_RE = re.compile(r"^[0-9a-fA-F]{2}(?::[0-9a-fA-F]{2}){5}$")
SPEEDS = {"25G": ("BF_SPEED_25G", 1), "100G": ("BF_SPEED_100G", 4)}
FECS = {"NONE": "BF_FEC_TYP_NONE", "RS": "BF_FEC_TYP_REED_SOLOMON", "FC": "BF_FEC_TYP_FIRECODE"}
logger = logging.getLogger("tofino_static_l2")


def parse_mac(value):
    if not MAC_RE.match(value):
        raise argparse.ArgumentTypeError("MAC must use six colon-separated octets")
    octets = value.split(":")
    if int(octets[0], 16) & 1:
        raise argparse.ArgumentTypeError("multicast MAC addresses are not supported")
    if int("".join(octets), 16) == 0:
        raise argparse.ArgumentTypeError("zero MAC address is not supported")
    return value.lower()


def build_desired_entries(dev_port_a, mac_a, dev_port_b, mac_b):
    """Return (ingress_dev_port, destination_mac_int) -> egress_dev_port."""
    if not (0 <= dev_port_a <= 511 and 0 <= dev_port_b <= 511):
        raise ValueError("dev_ports must be in the BFRT 9-bit port range")
    if dev_port_a == dev_port_b:
        raise ValueError("the two dev_ports must be different")
    if mac_a == mac_b:
        raise ValueError("the two endpoint MAC addresses must be different")
    return {
        (dev_port_a, int(mac_b.replace(":", ""), 16)): dev_port_b,
        (dev_port_b, int(mac_a.replace(":", ""), 16)): dev_port_a,
    }


def parse_ipv4(value):
    try:
        address = ipaddress.IPv4Address(value)
    except Exception:
        raise argparse.ArgumentTypeError("expected an IPv4 address")
    if address.is_multicast or address.is_unspecified or address.is_loopback:
        raise argparse.ArgumentTypeError("IPv4 address must be a unicast endpoint address")
    return str(address)


def build_desired_arp(ip_a, mac_a, ip_b, mac_b):
    if ip_a == ip_b:
        raise ValueError("the two endpoint IPv4 addresses must be different")
    if ipaddress.IPv4Network("{}/24".format(ip_a), strict=False) != ipaddress.IPv4Network(
        "{}/24".format(ip_b), strict=False
    ):
        raise ValueError("the two endpoints must be in the same /24 subnet")
    return {
        (1, int(ipaddress.IPv4Address(ip_a))): int(mac_a.replace(":", ""), 16),
        (1, int(ipaddress.IPv4Address(ip_b))): int(mac_b.replace(":", ""), 16),
    }


def _field_value(value):
    return value.get("value") if isinstance(value, dict) else value


def _connect(args):
    # Import SDE Python modules only for --apply so dry-run works on any host.
    import bfrt_grpc.client as gc

    client_id = args.client_id
    if client_id is None:
        client_id = (os.getpid() % 2000000000) + 1
    interface = gc.ClientInterface(
        grpc_addr=args.grpc_addr,
        client_id=client_id,
        device_id=args.device_id,
        perform_subscribe=True,
    )
    interface.bind_pipeline_config(args.program)
    info = interface.bfrt_info_get(args.program)
    target = gc.Target(device_id=args.device_id, pipe_id=args.pipe_id)
    return gc, interface, info, target


def _read_port_config(gc, info, target, dev_ports, speed, fec):
    """Read target ports and reject any pre-existing non-matching configuration."""
    expected_speed, expected_lanes = SPEEDS[speed]
    expected_fec = FECS[fec]
    table = info.table_get("$PORT")
    rows = list(table.entry_get(target, None, {"from_hw": True}))
    current = {}
    for data, key in rows:
        dev_port = _field_value(key.to_dict().get("$DEV_PORT"))
        if dev_port is not None and int(dev_port) in dev_ports:
            current[int(dev_port)] = data.to_dict()
    for dev_port in dev_ports:
        data = current.get(dev_port)
        if data is None:
            logger.info("dev_port=%s has no current $PORT entry", dev_port)
            continue
        actual = (data.get("$SPEED"), data.get("$N_LANES"), data.get("$FEC"))
        expected = (expected_speed, expected_lanes, expected_fec)
        if actual != expected:
            raise RuntimeError(
                "dev_port {} has unexpected speed/lanes/FEC: actual={} expected={}".format(
                    dev_port, actual, expected
                )
            )
    return current


def _configure_ports(gc, info, target, dev_ports, speed, fec, current):
    """Add missing port configs and enable exactly the selected dev_ports."""
    expected_speed, expected_lanes = SPEEDS[speed]
    expected_fec = FECS[fec]
    table = info.table_get("$PORT")
    missing = [dev_port for dev_port in dev_ports if dev_port not in current]
    if missing:
        keys = [table.make_key([gc.KeyTuple("$DEV_PORT", dev_port)]) for dev_port in missing]
        data = [table.make_data([
            gc.DataTuple("$SPEED", str_val=expected_speed),
            gc.DataTuple("$FEC", str_val=expected_fec),
            gc.DataTuple("$N_LANES", expected_lanes),
        ]) for _dev_port in missing]
        table.entry_add(target, keys, data)
    for dev_port in dev_ports:
        data = current.get(dev_port, {})
        if data.get("$PORT_ENABLE") and data.get("$PORT_UP"):
            continue
        key = [table.make_key([gc.KeyTuple("$DEV_PORT", dev_port)])]
        table.entry_mod(target, key, [table.make_data([
            gc.DataTuple("$PORT_ENABLE", bool_val=True),
            gc.DataTuple("$AUTO_NEGOTIATION", str_val="PM_AN_FORCE_DISABLE"),
        ])])
        logger.info("enabled target dev_port=%s only", dev_port)


def _wait_for_ports(gc, info, target, dev_ports, timeout_s):
    table = info.table_get("$PORT")
    deadline = time.time() + timeout_s
    while True:
        rows = list(table.entry_get(target, None, {"from_hw": True}))
        current = {}
        for data, key in rows:
            dev_port = _field_value(key.to_dict().get("$DEV_PORT"))
            if dev_port is not None and int(dev_port) in dev_ports:
                current[int(dev_port)] = data.to_dict()
        if all(
            current.get(dev_port, {}).get("$PORT_ENABLE")
            and current.get(dev_port, {}).get("$PORT_UP")
            for dev_port in dev_ports
        ):
            for dev_port in dev_ports:
                data = current[dev_port]
                logger.info("verified dev_port=%s name=%s speed=%s fec=%s UP", dev_port,
                            data.get("$PORT_NAME"), data.get("$SPEED"), data.get("$FEC"))
            return
        if time.time() >= deadline:
            state = {dev_port: current.get(dev_port) for dev_port in dev_ports}
            raise RuntimeError("target ports did not reach UP within {}s: {}".format(timeout_s, state))
        time.sleep(1)


def _verify_ports(gc, info, target, dev_ports):
    _wait_for_ports(gc, info, target, dev_ports, 1)


def _read_arp_entries(table, target, desired):
    current = {}
    wanted = set(desired)
    for data, key in table.entry_get(target, None, {"from_hw": False}):
        key_dict = key.to_dict()
        opcode = _field_value(key_dict.get("hdr.arp.opcode"))
        target_ip = _field_value(key_dict.get("hdr.arp.tpa"))
        if opcode is None or target_ip is None:
            continue
        managed_key = (int(opcode), int(target_ip))
        if managed_key not in wanted:
            continue
        mac = _field_value(data.to_dict().get("target_mac"))
        if mac is None:
            raise RuntimeError("managed ARP entry {} has no target_mac".format(managed_key))
        current[managed_key] = int(mac)
    return current


def _reconcile_arp(gc, info, target, desired, ip_by_key, mac_by_int):
    table = info.table_get(DEFAULT_ARP_TABLE)
    current = _read_arp_entries(table, target, desired)
    conflicts = {key: value for key, value in current.items() if value != desired[key]}
    if conflicts:
        raise RuntimeError("refusing to overwrite conflicting proxy_arp entries: {}".format(conflicts))
    missing = [key for key in desired if key not in current]
    keys = [table.make_key([
        gc.KeyTuple("hdr.arp.opcode", opcode),
        gc.KeyTuple("hdr.arp.tpa", target_ip),
    ]) for opcode, target_ip in missing]
    data = [table.make_data([
        gc.DataTuple("target_mac", desired[key]),
    ], "SwitchIngress.proxy_arp_reply") for key in missing]
    if missing:
        table.entry_add(target, keys, data)
    verified = _read_arp_entries(table, target, desired)
    if verified != desired:
        raise RuntimeError("proxy_arp readback mismatch: expected={} actual={}".format(desired, verified))
    for key in desired:
        if key in missing:
            print("add proxy_arp {} -> {}".format(ip_by_key[key], mac_by_int[desired[key]]))
        else:
            print("keep proxy_arp {} -> {}".format(ip_by_key[key], mac_by_int[desired[key]]))


def _read_managed_entries(table, target, desired):
    wanted = set(desired)
    current = {}
    for data, key in table.entry_get(target, None, {"from_hw": False}):
        key_dict = key.to_dict()
        ingress = _field_value(key_dict.get(INGRESS_PORT_KEY))
        destination = _field_value(key_dict.get(DEST_MAC_KEY))
        if ingress is None or destination is None:
            continue
        managed_key = (int(ingress), int(destination))
        if managed_key not in wanted:
            continue
        egress = _field_value(data.to_dict().get("port"))
        if egress is None:
            raise RuntimeError("managed forwarding entry {} has no egress port".format(managed_key))
        current[managed_key] = int(egress)
    return current


def _reconcile(gc, info, target, desired, table_name):
    table = info.table_get(table_name)
    current = _read_managed_entries(table, target, desired)

    conflicts = {
        key: value for key, value in current.items()
        if value != desired[key]
    }
    if conflicts:
        raise RuntimeError("refusing to overwrite conflicting entries: {}".format(conflicts))

    missing = [key for key in desired if key not in current]
    for (ingress, destination), egress in desired.items():
        print(
            "{} ingress={} dst_mac={:012x} -> egress={}".format(
                "add" if (ingress, destination) in missing else "keep",
                ingress,
                destination,
                egress,
            )
        )
    if not missing:
        if current != desired:
            raise RuntimeError("forwarding readback mismatch: expected={} actual={}".format(desired, current))
        return

    keys = [
        table.make_key([
            gc.KeyTuple(INGRESS_PORT_KEY, ingress),
            gc.KeyTuple(DEST_MAC_KEY, destination),
        ])
        for ingress, destination in missing
    ]
    data = [
        table.make_data([gc.DataTuple("port", desired[key])], "SwitchIngress.set_egress")
        for key in missing
    ]
    table.entry_add(target, keys, data)

    verified = _read_managed_entries(table, target, desired)
    if verified != desired:
        raise RuntimeError(
            "forwarding readback mismatch: expected={} actual={}".format(desired, verified)
        )
    logger.info("verified all requested forwarding entries after add")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Add two exact-match L2 entries to an already active Tofino pipeline"
    )
    parser.add_argument("--grpc-addr", default="localhost:50052")
    parser.add_argument("--program", default=DEFAULT_PROGRAM)
    parser.add_argument("--forward-table", default=DEFAULT_FORWARD_TABLE)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--pipe-id", type=lambda value: int(value, 0), default=0xFFFF)
    parser.add_argument("--client-id", type=int)
    parser.add_argument("--dev-port-a", type=int, required=True)
    parser.add_argument("--ip-a", type=parse_ipv4, required=True)
    parser.add_argument("--mac-a", type=parse_mac, required=True)
    parser.add_argument("--dev-port-b", type=int, required=True)
    parser.add_argument("--ip-b", type=parse_ipv4, required=True)
    parser.add_argument("--mac-b", type=parse_mac, required=True)
    parser.add_argument("--speed", choices=sorted(SPEEDS), default="100G")
    parser.add_argument("--fec", choices=sorted(FECS), default="NONE")
    parser.add_argument("--configure-ports", action="store_true",
                        help="add/enable only the two requested ports after read-only preflight")
    parser.add_argument("--link-timeout", type=int, default=60)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", dest="apply", action="store_false")
    mode.add_argument("--apply", dest="apply", action="store_true")
    parser.set_defaults(apply=False)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        desired = build_desired_entries(
            args.dev_port_a, args.mac_a, args.dev_port_b, args.mac_b
        )
        desired_arp = build_desired_arp(args.ip_a, args.mac_a, args.ip_b, args.mac_b)
    except ValueError as exc:
        logger.error("invalid endpoint pair: %s", exc)
        return 2

    if not args.apply:
        print("DRY RUN: no BFRT connection or switch changes")
        print("configure_ports={} speed={} fec={}".format(args.configure_ports, args.speed, args.fec))
        for ip_address, mac_address in ((args.ip_a, args.mac_a), (args.ip_b, args.mac_b)):
            print("proxy_arp {} -> {}".format(ip_address, mac_address))
        for (ingress, destination), egress in desired.items():
            print("add ingress={} dst_mac={:012x} -> egress={}".format(
                ingress, destination, egress
            ))
        return 0

    interface = None
    try:
        gc, interface, info, target = _connect(args)
        dev_ports = (args.dev_port_a, args.dev_port_b)
        current_ports = _read_port_config(gc, info, target, dev_ports, args.speed, args.fec)
        # Refuse table conflicts before touching even the two target port settings.
        forward_table = info.table_get(args.forward_table)
        forward_current = _read_managed_entries(forward_table, target, desired)
        forward_conflicts = {key: value for key, value in forward_current.items()
                             if value != desired[key]}
        if forward_conflicts:
            raise RuntimeError("refusing to overwrite conflicting forwarding entries: {}".format(
                forward_conflicts))
        arp_table = info.table_get(DEFAULT_ARP_TABLE)
        arp_current = _read_arp_entries(arp_table, target, desired_arp)
        arp_conflicts = {key: value for key, value in arp_current.items()
                         if value != desired_arp[key]}
        if arp_conflicts:
            raise RuntimeError("refusing to overwrite conflicting proxy_arp entries: {}".format(
                arp_conflicts))
        if args.configure_ports:
            _configure_ports(gc, info, target, dev_ports, args.speed, args.fec, current_ports)
            _wait_for_ports(gc, info, target, dev_ports, args.link_timeout)
        else:
            _verify_ports(gc, info, target, dev_ports)
        ip_by_key = {
            (1, int(ipaddress.IPv4Address(args.ip_a))): args.ip_a,
            (1, int(ipaddress.IPv4Address(args.ip_b))): args.ip_b,
        }
        mac_by_int = {
            int(args.mac_a.replace(":", ""), 16): args.mac_a,
            int(args.mac_b.replace(":", ""), 16): args.mac_b,
        }
        _reconcile_arp(gc, info, target, desired_arp, ip_by_key, mac_by_int)
        _reconcile(gc, info, target, desired, args.forward_table)
        return 0
    except Exception as exc:
        logger.error("apply failed: %s", exc)
        return 1
    finally:
        if interface is not None:
            try:
                interface.tear_down_stream()
            except Exception:
                pass
            try:
                interface.channel.close()
            except Exception:
                pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.exit(main())
