# MATHIR for Robotics — Place Memory as a Complement to Sensor Fusion

**Status:** research direction, pre-simulation. This is the seed document for a dedicated **MATHIR FOR ROBOTICS** track/repo. It states a hypothesis, grounds it in the existing literature, and is explicit about what is not yet tested.

---

## 1. The question this track is trying to answer

Autonomous vehicles depend on sensors — camera, LiDAR, radar, IMU. Three concrete scenarios motivate this work:

1. A sensor is disconnected or fails mid-drive (e.g. 5 of 8 sensors go down). Does the system know what to do, or does it silently degrade?
2. Something falls off the vehicle and the sensors don't detect it. Is there any record of what happened, or is it lost the instant the frame passes?
3. The vehicle avoids an obstacle (a rock) using its sensors on a stretch of road. It is driven over the *same* stretch again. Does anything carry over, or does it start from zero every time?

The hypothesis: a **persistent, place-based episodic memory** — MATHIR sitting between perception and decision — can act as a fallback signal when sensor confidence collapses, and can retain experience (what happened at this location, what decision was taken, what the outcome was) independently of which sensors produced it.

This is **not** a claim that MATHIR replaces LiDAR, cameras, or sensor fusion. A vehicle still needs sensors to perceive. The goal is to reduce over-dependence on any single sensor snapshot and give the system something to fall back on that isn't "guess from degraded data."

---

## 2. Where this sits relative to existing work (honest positioning)

### 2.1 Sensor-dropout robustness — an active, well-studied field

"What happens if you unplug a sensor mid-drive" is a named problem: **graceful degradation under sensor dropout**. Recent work (Grace-BEV, MetaBEV, UniBEV, RCTrans, CramNet) tests exactly this — drop 1, 3, 6 sensors and measure mAP/NDS on standard benchmarks (nuScenes-R, nuScenes-C). Grace-BEV (2026) restores up to 34.7% accuracy under catastrophic LiDAR failure where classical fusion collapses to 0%.

**What this field does not do:** none of it uses memory of a specific place. These systems are robust to missing data *at every instant, independent of location history* — they don't remember "this stretch of road" at all. That's the gap MATHIR's hypothesis targets.

**A real limitation in that field, not yet solved elsewhere:** models trained on one sensor configuration (LiDAR beam count, camera FOV/position, resolution) degrade when the configuration changes — even across vehicles in the same fleet with different mounting positions. This is described in 2025 literature as "largely unexplored." If MATHIR stores *abstracted experience* ("obstacle of type X at this location, decision Y taken, outcome Z") rather than raw sensor signatures, that memory is in principle sensor-agnostic — but only as far as perception correctly abstracted the raw signal in the first place. MATHIR does not fix perception robustness; it assumes perception already did its job at the moment the memory was written. This is an explicit limitation, not a hidden assumption.

### 2.2 Place-based memory — exists, but for offline maps, not real-time fallback

Crowdsourced road experience already exists in production (Mobileye REM, Tesla fleet learning) — but for building offline HD maps, not as a real-time fallback mechanism when sensors fail mid-drive. No published work combines the two: using episodic place-memory as a fallback signal precisely when sensor-fusion confidence collapses. That combination is the specific, testable gap this track targets.

### 2.3 LLM-in-the-loop driving — latency is the documented blocker

A field called **LLM4AD** puts LLMs/VLMs directly in driving decision loops. Their own numbers: the most responsive LLMs still take 1.2–1.8s per reasoning step — too slow to reason every timestep, and reasoning based on a stale world-state has caused failures in their own simulations. This is a real, named, unsolved problem, not a MATHIR insight.

MATHIR's angle is different from "make each inference call faster" (what model vendors — e.g. DeepSeek's DSpark speculative decoding, 60–85% average throughput gains — are already doing, and which MATHIR benefits from passively, not by merit). MATHIR's contribution, if validated, is reducing **how often** a full LLM reasoning call is needed: if the system recognizes "I've seen a very similar situation here before," it can act on that recognition (a vector lookup, tens of ms per MATHIR's own reranking benchmarks) instead of triggering a full ~1.2–1.8s reasoning pass. That is a frequency reduction, not a per-call speedup — a different lever than what the LLM4AD or inference-speed literature is optimizing.

**Important caveat, stated explicitly:** even MATHIR's own lookup latency (~tens–100ms on CPU, per current benchmarks) is far too slow to be the real-time control loop itself (steering/motor control runs at tens of Hz — a few ms budget). The architecture must stay: **MATHIR + any LLM = advisory/recognition layer feeding a classical real-time controller (PID/MPC), never the control loop itself.**

---

## 3. What is and isn't validated today

**Tested and working (software, general-purpose, not driving-specific):** 6-tier memory architecture, decay/promotion/consolidation, cross-process multi-agent sharing (God Mode) across multiple LLM providers, fully local with no cloud dependency.

**Not yet tested:** the place-memory-as-sensor-fallback hypothesis itself. No simulation run, no RC car run, no collision-rate/safety-distance numbers exist yet for this specific claim.

**Hardware honesty:** all current MATHIR numbers come from consumer hardware (laptop CPU), not automotive-grade or industrial embedded hardware. The edge path (Raspberry Pi, Jetson) is a planned next step, not a completed validation.

---

## 4. Proposed protocol (next step: simulation, then RC car)

1. Drive a fixed course multiple times with all sensors active — MATHIR records the experience (obstacles, decisions, outcomes).
2. Repeat the same course with N of M sensors deliberately disabled.
3. Compare three conditions: (a) degraded sensors alone, no MATHIR; (b) a simple sensor-dropout-robust baseline (training-time dropout, not a full Grace-BEV reimplementation); (c) MATHIR injecting "I've seen this stretch before, here's what it contains" as a complement to (b).
4. Metrics fixed **before** running, not chosen after seeing results: collision rate, minimum safe distance, reaction time, and — specific to MATHIR's own claim — frequency of full LLM reasoning calls with vs. without MATHIR.
5. Log everything, including failures.

**Sample-size honesty:** one RC car, one course = a small sample. This supports "proof of concept on a controlled scenario," not a generalizable industry claim.

---

## 5. Why this is a separate track from core MATHIR

Core MATHIR (this repo) is a general-purpose memory layer for LLM agents/chat/dev tools — validated in software, general-purpose, not domain-specific. The robotics hypothesis above is domain-specific, unvalidated, and carries different risk/rigor requirements (safety-critical framing, simulation-first methodology). Keeping them separate avoids overloading the general memory-layer pitch with an unproven, ambitious research claim, and lets each be evaluated on its own merits.
