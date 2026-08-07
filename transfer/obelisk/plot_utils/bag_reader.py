# Copyright 2026 Zachary Olkin. All rights reserved.

"""Read an obelisk MCAP run, with no ROS installation and no finalised bag.

Two properties matter here and neither is free:

**No ROS.** MCAP embeds every message's schema in the file, so ``mcap`` +
``mcap-ros2-support`` decode ``g1_msgs/ControllerDebug`` on a laptop that has never heard of
this workspace. That is the whole reason the recorder writes MCAP rather than sqlite3.

**No finalised bag.** rosbag2 writes ``metadata.yaml`` and the MCAP summary only on a clean
shutdown, and a run that ended in a fall usually did not get one. Two consequences, both
handled here:

* the INDEXED reader needs the summary and dies on its absence
  (``RecordLengthLimitExceeded``, reading garbage past the end), so this uses the streaming
  reader with ``log_time_order=False`` -- a pure linear scan that needs no index and no
  buffering;
* the last record is usually torn mid-write, so the scan stops at the tear and keeps
  everything before it. A killed 69 MB bag recovers ~61 000 messages this way.

``rosbag2_py`` is the in-container fallback, for a bag that HAS been finalised or reindexed.
"""

from __future__ import annotations

import glob
import os
import sys
from collections import defaultdict


def mcap_files(bag: str) -> list[str]:
    """The ``.mcap`` files of a bag, whether given as a directory or a single file.

    Args:
        bag: Bag directory or ``.mcap`` path.

    Returns:
        Sorted ``.mcap`` paths (a long recording is split across several).
    """
    if os.path.isfile(bag):
        return [bag]
    return sorted(glob.glob(os.path.join(bag, "*.mcap")))


def read_bag(bag: str, topics: set[str] | None = None) -> dict[str, list]:
    """Read a run into ``{topic: [(t_seconds, message), ...]}``.

    Args:
        bag: Bag directory or ``.mcap`` file.
        topics: Topics to keep; None keeps everything.

    Returns:
        Per-topic message lists in log order.

    Raises:
        SystemExit: If the bag holds no ``.mcap``, or neither reader is importable.
    """
    files = mcap_files(bag)
    if not files:
        sys.exit(f"[ERROR] No .mcap files in {bag}. Was it recorded with `-s mcap`?")
    try:
        from mcap.reader import NonSeekingReader
        from mcap_ros2.decoder import DecoderFactory
    except ImportError:
        return _read_rosbag2(bag, topics)

    out: dict[str, list] = defaultdict(list)
    for path in files:
        with open(path, "rb") as fh:
            reader = NonSeekingReader(fh, decoder_factories=[DecoderFactory()])
            # log_time_order=False is load-bearing, not a micro-optimisation: the ordered
            # path sorts, which buffers the WHOLE file before yielding anything -- so a tear
            # at the end loses every message rather than the last few.
            it = reader.iter_decoded_messages(
                topics=sorted(topics) if topics else None, log_time_order=False)
            n = 0
            while True:
                try:
                    _, channel, message, ros_msg = next(it)
                except StopIteration:
                    break
                except Exception as exc:                              # noqa: BLE001
                    print(f"[WARN] {os.path.basename(path)} is torn at message ~{n} "
                          f"({type(exc).__name__}) -- the recorder was killed mid-write. "
                          f"Everything before the tear was recovered; `stop.sh` shuts the "
                          f"recorder down gracefully now.")
                    break
                out[channel.topic].append((message.log_time * 1e-9, ros_msg))
                n += 1
    # A linear scan follows chunk order, which is log order per writer but not guaranteed
    # across chunks; sort so downstream can assume monotonic time.
    for topic in out:
        out[topic].sort(key=lambda item: item[0])
    return out


def _read_rosbag2(bag: str, topics: set[str] | None) -> dict[str, list]:
    """In-container fallback: ``rosbag2_py`` plus the built message packages.

    Needs a FINALISED bag (``metadata.yaml``); run ``ros2 bag reindex <bag> -s mcap`` first
    if the recorder was killed. Prefer the mcap path, which needs neither.

    Args:
        bag: Bag directory.
        topics: Topics to keep; None keeps everything.

    Returns:
        Same structure as :func:`read_bag`.

    Raises:
        SystemExit: If rosbag2_py is unavailable too.
    """
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError:
        sys.exit(
            "[ERROR] Neither mcap nor rosbag2_py is importable.\n"
            "  Off-container (no ROS needed -- MCAP carries the schemas):\n"
            "    pip install mcap mcap-ros2-support matplotlib numpy\n"
            "  In-container: source the workspace first (obk; source install/setup.bash)."
        )
    print(
        "[WARN] falling back to rosbag2_py, which deserialises with the INSTALLED message\n"
        "       definition rather than the one stored in the bag. A bag recorded before a\n"
        "       .msg change is then read through the wrong layout and every field after the\n"
        "       edit silently shifts -- it does not error, it returns plausible numbers for\n"
        "       the wrong fields. (This exact thing reported 100 stale depth ticks that were\n"
        "       really the removed `warming_up` flag: 2.0 s of warm-up at 50 Hz.)\n"
        "       Prefer the mcap reader, which uses the bag's own schema:\n"
        "         pip install mcap mcap-ros2-support"
    )
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag, storage_id="mcap"),
                rosbag2_py.ConverterOptions("", ""))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    out: dict[str, list] = defaultdict(list)
    while reader.has_next():
        topic, data, stamp = reader.read_next()
        if topics is None or topic in topics:
            out[topic].append((stamp * 1e-9, deserialize_message(data, get_message(types[topic]))))
    return out
