# UX feedback from real gauntlet use (2026-07-04)

Written after driving the tool end to end against real RunPod: deploying qwen3-0.6b and qwen3-32b,
running the validation gauntlet (kill-pod, crash-recovery, orphan sweep, concurrent deploys +
proxy routing), and hitting three real-API bugs along the way (ports array, model_ready id, proxy
model rewrite). This is from use, not review.

## What works well

- **The reconciler earns its keep under real chaos.** Killing a pod out of band, simulating a crash
  where the instance id was never persisted, and leaving orphan pods around all recovered with no
  special-case code, just the desired/observed loop. The architecture paid off where it counts.
- **Cost-safety is real, not a slogan.** Every failed deploy left zero pods, verified repeatedly.
  The orphan sweep with namespace isolation is a genuinely good touch for the scariest failure mode
  in GPU tooling (a pod billing forever).
- **`gpu status` / `gpu costs` with live accrual is the right thing to surface.** Watching the dollar
  figure per deployment tick up is exactly what a user wants.

## What frustrated me (in priority order)

1. **Cold start is slow and invisible. The number-one problem.** qwen3-0.6b took 5 to 8 minutes;
   qwen3-32b took ~17 to 40. During all of it, `gpu status` just says `starting_server` with no
   signal: no "pulling image", no "downloaded 12GB of 65GB", no ETA. I ended up curling the pod's
   `/health` and `/v1/models` directly to understand progress, which is the tool's job. The
   `DOWNLOADING` state exists in the model but collapses into `STARTING` because RunPod's REST API
   exposes no pod logs to parse progress from. Staring at a billing A100 with no idea if it is 20%
   or 90% done is genuinely stressful.

2. **Re-downloading the model every time is the real slowness.** Ephemeral disks mean qwen3-32b
   pulls 65GB on every deploy. That is what makes the tool feel slow, more than anything in the
   engine. The `VolumeSpec` seam already exists for a persistent cache. Turning a 40-minute cold
   start into a 2-minute warm one is higher leverage than growing the catalog.

3. **Non-blocking deploy silently does nothing without a daemon.** `gpu deploy` without `--wait` and
   with no `gpu daemon` running writes the record and never creates a pod, with no warning. That is
   a footgun. The two blocking foreground processes (`gpu daemon`, `gpu proxy`) also mean real
   background operation requires manual process management. There is no `gpu up`.

4. **"Zero to chat in 5 minutes" is aspirational today.** Realistically 8-plus minutes for the
   smallest model, and "chat" still means deploy, wait, start the proxy in another terminal, then
   hit it. No single `gpu deploy qwen3-0.6b --chat` deploys, waits, and drops you into a REPL.

5. **The catalog's `validated_at` was fiction until this session.** The metadata claimed a
   2026-07-03 validation, but the three real-API bugs found here prove nothing had touched real
   RunPod before. qwen3-0.6b and qwen3-32b are now genuinely validated; the metadata should be
   regenerated from real runs, and "validated" should mean "we ran it".

## Summary

The engine is strong and the hard correctness problems (recovery, cost-safety) are solved well. The
gap between "impressive engine" and "delightful product" is almost entirely last-mile UX around
cold-start visibility and the background lifecycle, not anything deep.

The top three fixes are planned in [plan-ux-improvements.md](plan-ux-improvements.md):
1. Download progress + ETA in `gpu status`.
2. Persistent model cache volume.
3. Make the daemon lifecycle invisible (never let a non-blocking deploy silently stall).
