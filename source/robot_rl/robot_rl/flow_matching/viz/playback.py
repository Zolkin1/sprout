# Copyright 2026 Zachary Olkin. All rights reserved.

"""Playback GUI (play/pause, frame stepping, speed) for the viser viewer.

All frame changes funnel through the frame slider: the play loop, the
prev/next buttons, and direct dragging all set ``frame_slider.value``, whose
``on_update`` callback drives the viewer's single ``set_frame`` code path.

viser has no keyboard-event API, so arrow-key stepping is the browser-native
behavior of a focused slider: click the frame slider once, then use the
arrow keys.
"""

from __future__ import annotations

import time
from typing import Callable

import viser

_PAUSED_POLL_PERIOD_S = 0.05


class PlaybackControls:
    """Play/pause + frame-stepping GUI and the main-thread playback loop."""

    def __init__(self, server: viser.ViserServer, num_frames: int, dt: float):
        """Create the "Playback" GUI folder.

        Args:
            server: The viser server.
            num_frames: Total recorded frames T.
            dt: Recorded control step period in seconds (playback rate at 1x).
        """
        self.num_frames = num_frames
        self.dt = dt
        self.playing = False

        with server.gui.add_folder("Playback"):
            self._play_button = server.gui.add_button("Play", hint="Loop through the recording at the recorded dt.")
            self._prev_button = server.gui.add_button("Prev frame")
            self._next_button = server.gui.add_button("Next frame")
            self.frame_slider = server.gui.add_slider(
                "Frame", min=0, max=num_frames - 1, step=1, initial_value=0,
                hint="Click the slider, then use the arrow keys to step one frame at a time.",
            )
            self._speed_slider = server.gui.add_slider(
                "Speed", min=0.25, max=4.0, step=0.25, initial_value=1.0, hint="Playback speed multiplier."
            )
            self._readout = server.gui.add_markdown(self._readout_text(0))

        @self._play_button.on_click
        def _(_) -> None:
            self.playing = not self.playing
            self._play_button.label = "Pause" if self.playing else "Play"

        @self._prev_button.on_click
        def _(_) -> None:
            self.frame_slider.value = max(0, self.frame_slider.value - 1)

        @self._next_button.on_click
        def _(_) -> None:
            self.frame_slider.value = min(self.num_frames - 1, self.frame_slider.value + 1)

    def _readout_text(self, frame: int) -> str:
        """Markdown for the frame-number readout."""
        return f"**Frame {frame} / {self.num_frames - 1}** — t = {frame * self.dt:.2f} s"

    def on_frame(self, fn: Callable[[int], None]) -> None:
        """Register the frame callback; also keeps the readout in sync."""

        @self.frame_slider.on_update
        def _(_) -> None:
            frame = int(self.frame_slider.value)
            self._readout.content = self._readout_text(frame)
            fn(frame)

    def run(self) -> None:
        """Main-thread loop: advance the slider while playing (loops at the end)."""
        while True:
            t0 = time.perf_counter()
            if self.playing:
                self.frame_slider.value = (self.frame_slider.value + 1) % self.num_frames
                period = self.dt / self._speed_slider.value
            else:
                period = _PAUSED_POLL_PERIOD_S
            time.sleep(max(0.0, period - (time.perf_counter() - t0)))
