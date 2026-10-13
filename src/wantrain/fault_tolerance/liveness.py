"""Failure detection: convict a dead worker fast, and only the right one.

Signals, in rising cost (ported from the source orchestrator, which probed
rented boxes over ssh; here every worker agent heartbeats over its control
connection and reports its stage process):

  S1  the agent's control connection is gone and does not come back within
      the reconnect grace (WAN blips reconnect; dead hosts do not);
  S2  a slot's stage process died without the engine's done marker (killed
      from outside: OOM killer, host reboot, container restart) -- convicts
      at once, it is a fact, not a suspicion;
  S3  a slot exited with "peer connection lost" while another is silent: the
      silent one is the suspect;
  S4  no optimizer step fleet-wide in max(floor, 6 x median step): every
      slot is examined. S4 judges only while some slot still owes an
      optimizer step, from WT_STEP progress inside [the attempt's start step,
      the final step]; once every slot finished the final step, quiet is
      shutdown (final wave, loss gather, drain, last ship), bounded by
      `shutdown_limit_s`, never a stall.

A heartbeat gap is a first strike; conviction needs a second look at least
`strike_gap_s` later, because a proxy blink looks the same as a dead box.
Exactly one convicted slot is healed at a time: several at once usually point
at the coordinator's own network, and replacing hardware would not fix that.
If every worker is alive and the fleet still makes no progress, the data plane
between them is broken (a WAN partition), which is recovered by restarting in
place from the last checkpoint rather than by replacing anyone.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field

STALL_FLOOR_S = 90.0
STALL_STEPS = 6
STRIKE_GAP_S = 8.0
HEARTBEAT_TIMEOUT_S = 15.0
RECONNECT_GRACE_S = 30.0
SHUTDOWN_LIMIT_S = 600.0

PEER_LOST = "peer connection lost"
DONE_MARKER = "WT_DONE"


@dataclass
class SlotState:
    """What the coordinator knows about one (rank, peer) slot right now."""

    slot: tuple
    node_id: str
    launched: bool = False
    connected: bool = True
    disconnected_at: float | None = None
    last_heartbeat: float = 0.0
    stage_alive: bool | None = None
    exit_code: int | None = None
    done_marker: bool = False  # the engine printed its done marker
    log_tail: str = ""
    progress: list = field(default_factory=list)  # [(step, t)] from the worker
    suspect_at: float | None = None
    started_at: float = 0.0
    gone_since: float | None = None  # first time the stage was seen gone, no exit yet
    peer_lost_at: float | None = None  # first time the judge saw this slot's "peer lost" exit
    start_step: int = 0  # first optimizer step this attempt runs (wave + 1 on a resume)

    def last_step(self) -> int | None:
        return self.progress[-1][0] if self.progress else None

    def add_progress(self, step: int, t: float, final_step: int | None) -> bool:
        """Record an optimizer step this attempt ran; anything outside
        [start_step, final_step] or not strictly greater than the last
        recorded step is not progress (ADR-0003: a replayed or looping stream
        never renews the stall clock, and `progress` stays bounded)."""
        if step < self.start_step or (final_step is not None and step > final_step):
            return False
        if self.progress and step <= self.progress[-1][0]:
            return False
        self.progress.append((step, t))
        return True


@dataclass
class Verdict:
    kind: str  # ok | suspect | convict | stalled | failed | done
    slot: tuple | None = None
    why: str = ""
    slots: list = field(default_factory=list)


def median_step_s(states) -> float | None:
    """Median gap between the fleet's recent optimizer steps."""
    pts = sorted((s, t) for st in states for (s, t) in st.progress)
    gaps = []
    for (s0, t0), (s1, t1) in zip(pts, pts[1:]):
        if s1 > s0 and t1 > t0:
            gaps.append((t1 - t0) / (s1 - s0))
    if len(gaps) < 3:
        return None
    return statistics.median(gaps[-12:])


def last_progress_at(states) -> float | None:
    ts = [t for st in states for (_s, t) in st.progress]
    if ts:
        return max(ts)
    started = [st.started_at for st in states if st.launched and st.started_at]
    return max(started) if started else None


class LivenessJudge:
    def __init__(
        self,
        heartbeat_timeout_s: float = HEARTBEAT_TIMEOUT_S,
        reconnect_grace_s: float = RECONNECT_GRACE_S,
        stall_floor_s: float = STALL_FLOOR_S,
        stall_steps: int = STALL_STEPS,
        strike_gap_s: float = STRIKE_GAP_S,
        first_step_timeout_s: float = 600.0,
        exit_report_grace_s: float = 10.0,
        final_step: int | None = None,
        shutdown_limit_s: float = SHUTDOWN_LIMIT_S,
    ):
        self.heartbeat_timeout_s = heartbeat_timeout_s
        self.reconnect_grace_s = reconnect_grace_s
        self.stall_floor_s = stall_floor_s
        self.stall_steps = stall_steps
        self.strike_gap_s = strike_gap_s
        self.first_step_timeout_s = first_step_timeout_s
        # A heartbeat can say "the stage is gone" a moment before the exit
        # report that explains it arrives; only a gone stage that stays
        # unexplained this long was killed without its supervisor noticing.
        self.exit_report_grace_s = exit_report_grace_s
        # The run's last optimizer step; None judges without a shutdown phase.
        self.final_step = final_step
        self.shutdown_limit_s = shutdown_limit_s

    def _in_shutdown(self, launched) -> bool:
        """Every launched slot finished the final step: it owes no optimizer step."""
        f = self.final_step
        if f is None:
            return False
        return all(st.exit_code == 0 or st.last_step() == f or st.start_step > f for st in launched)

    # ------------------------------------------------------------ classification
    def _process_dead(self, st: SlotState, now: float) -> str | None:
        """A stage that ended without its done marker was killed from outside."""
        if st.exit_code is None:
            if st.stage_alive is False and st.launched and st.connected:
                if st.gone_since is None:
                    st.gone_since = now
                if now - st.gone_since > self.exit_report_grace_s:
                    return "the stage process is gone and no exit was reported"
            else:
                st.gone_since = None
            return None
        if st.exit_code != 0 and not st.done_marker:
            return (
                f"the stage process died with code {st.exit_code} and no done marker "
                f"(killed from outside: OOM killer, reboot, restart)"
            )
        return None

    def _unreachable(self, st: SlotState, now: float) -> str | None:
        """Heartbeat silence on a connection that is still open. A closed
        connection is judged by the reconnect grace (S1), not by strikes."""
        if not st.connected:
            return None
        if st.last_heartbeat and now - st.last_heartbeat > self.heartbeat_timeout_s:
            return f"no heartbeat for {now - st.last_heartbeat:.0f}s"
        return None

    def judge(self, states, now: float | None = None) -> Verdict:
        now = time.time() if now is None else now
        launched = [st for st in states if st.launched]
        if not launched:
            return Verdict("ok")

        # Done: every slot exited cleanly.
        if all(st.exit_code == 0 for st in launched):
            return Verdict("done")

        # S2: a process that died from outside convicts at once.
        dead = [(st, why) for st in launched if (why := self._process_dead(st, now))]
        # S1: lost control connections past the reconnect grace.
        gone = []
        for st in launched:
            if st.exit_code is not None:
                continue
            if (
                not st.connected
                and st.disconnected_at is not None
                and now - st.disconnected_at > self.reconnect_grace_s
            ):
                gone.append(
                    (
                        st,
                        f"worker unreachable: control connection lost "
                        f"{now - st.disconnected_at:.0f}s ago and not re-established",
                    )
                )
        convicted = dead + gone
        if len(convicted) == 1:
            st, why = convicted[0]
            return Verdict("convict", st.slot, why)
        if len(convicted) > 1:
            return Verdict(
                "failed",
                None,
                f"{len(convicted)} slots failed at once "
                f"({', '.join(str(s.slot) for s, _ in convicted)}): that is not one "
                f"dead worker, it is a network or coordinator problem, and replacing "
                f"hardware would not fix it",
                [s.slot for s, _ in convicted],
            )

        # Inside the reconnect grace nothing else is judged: a WAN blip that
        # heals itself must not trigger a restart or a conviction.
        waiting = [st for st in launched if st.exit_code is None and not st.connected]
        if waiting:
            st = waiting[0]
            since = now - (st.disconnected_at or now)
            return Verdict(
                "suspect",
                st.slot,
                f"control connection lost {since:.0f}s ago; waiting up to "
                f"{self.reconnect_grace_s:.0f}s for the worker to reconnect",
                [w.slot for w in waiting],
            )

        # A deterministic error (an exception, an OOM the engine reported) is
        # not a dead box: a replacement would hit it again.
        crashed = [
            st
            for st in launched
            if st.exit_code not in (None, 0) and st.done_marker and PEER_LOST not in st.log_tail
        ]
        if crashed:
            st = crashed[0]
            return Verdict(
                "failed",
                st.slot,
                f"slot {st.slot} raised (exit {st.exit_code}); this is the run, not "
                f"the hardware. Log tail: {st.log_tail[-300:]}",
            )

        # S3 / S4: who to look at.
        peer_lost = any(
            st.exit_code not in (None, 0) and PEER_LOST in st.log_tail for st in launched
        )
        silent = [st for st in launched if st.exit_code is None]
        step_s = median_step_s(launched)
        seen = last_progress_at(launched)
        if step_s is None:
            limit = self.first_step_timeout_s
        else:
            limit = max(self.stall_floor_s, self.stall_steps * step_s)
        if self._in_shutdown(launched):
            # No step is owed: quiet is shutdown, bounded, never a stall.
            ts = [t for st in launched for (_s, t) in st.progress]
            ts += [st.started_at for st in launched if st.started_at]
            quiet = now - max(ts) if ts else 0.0
            # S3 judges a silent slot first; with no silent slot left (every
            # stage exited) only the bound ends the attempt.
            if not (peer_lost and silent) and quiet > self.shutdown_limit_s:
                return Verdict(
                    "failed",
                    None,
                    f"shutdown exceeded {self.shutdown_limit_s:.0f}s after the final step "
                    f"(quiet {quiet:.0f}s with every slot past step {self.final_step}); "
                    f"a large final wave needs a higher shutdown_limit_s",
                    [st.slot for st in launched],
                )
            stalled = False
        else:
            stalled = seen is not None and now - seen > limit
        suspects = silent if peer_lost else (launched if stalled else [])
        if not suspects:
            # Ordinary heartbeat staleness still earns a strike even without a stall.
            for st in launched:
                if st.exit_code is None and (why := self._unreachable(st, now)):
                    return self._strike(st, why, now)
            for st in launched:
                st.suspect_at = None
            return Verdict("ok")

        unreachable = [
            (st, why)
            for st in suspects
            if st.exit_code is None and (why := self._unreachable(st, now))
        ]
        if len(unreachable) == 1:
            st, why = unreachable[0]
            return self._strike(st, why, now)
        if len(unreachable) > 1:
            return Verdict(
                "suspect",
                None,
                f"{len(unreachable)} slots unreachable at once; waiting for the "
                f"reconnect grace before judging",
                [s.slot for s, _ in unreachable],
            )
        for st in suspects:
            st.suspect_at = None
        if peer_lost:
            # The dead slot's own EXIT travels on another worker's channel and
            # can land after its peer's "peer lost" exit. A silent slot gets
            # exit_report_grace_s to report (S2) before the fleet is "stalled".
            lost_at = []
            for st in launched:
                if st.exit_code not in (None, 0) and PEER_LOST in st.log_tail:
                    if st.peer_lost_at is None:
                        st.peer_lost_at = now
                    lost_at.append(st.peer_lost_at)
            if silent and now - min(lost_at) <= self.exit_report_grace_s:
                return Verdict(
                    "suspect",
                    None,
                    f"a stage lost its peer; waiting up to {self.exit_report_grace_s:.0f}s "
                    f"for the silent slots' exit reports",
                    [st.slot for st in silent],
                )
        # Everyone is alive and heartbeating, yet a peer was lost or nothing moves:
        # the data plane between them is broken.
        why = (
            "a stage lost its peer while every worker is alive"
            if peer_lost
            else f"no optimizer step fleet-wide for {now - (seen or now):.0f}s "
            f"(limit {limit:.0f}s) while every worker is alive"
        )
        return Verdict("stalled", None, why, [st.slot for st in launched])

    def _strike(self, st: SlotState, why: str, now: float) -> Verdict:
        if st.suspect_at is None:
            st.suspect_at = now
            return Verdict("suspect", st.slot, f"first strike: {why}")
        if now - st.suspect_at >= self.strike_gap_s:
            return Verdict(
                "convict",
                st.slot,
                f"{why}; still failing {now - st.suspect_at:.0f}s after the first strike",
            )
        return Verdict("suspect", st.slot, f"awaiting second strike: {why}")
