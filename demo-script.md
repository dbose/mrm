# asciinema demo script — riskattest 90-second walkthrough

Run `./demo-setup.sh` first. It seeds the chain, captures a
`DecisionRecord`, signs a Merkle root, and generates a real SR 26-2
report. ~5 seconds.

Then record with:

```bash
clear
asciinema rec --idle-time-limit=2 ~/riskattest-demo.cast
```

The `--idle-time-limit=2` flag automatically compresses any pause
longer than 2 seconds — your thinking-time becomes a 2-second beat in
playback, which is exactly what you want.

End the recording with `exit` or `Ctrl-D`.

## Pre-flight

Before pressing record, in the terminal:

1. **Terminal window: 80 cols × 24 rows.** asciinema embeds resize
   poorly — pick a standard size.
2. **Font: anything monospace, 14-16pt.**
3. **Prompt: short.** If yours is multi-line / fancy, temporarily
   set: `PS1='$ '`. Long prompts crowd the recording.
4. **Activate the env**: `conda activate mrm`.
5. **`cd` into the demo dir** so the relative paths work:
   `cd /Users/dbose/projects/mrm`.
6. **Set a shortcut alias** for legibility:
   `alias r='cat demo-artefacts/demo_record_id.txt'`. This lets you
   pipe the record-ID into commands without showing a long UUID
   on-screen typed character by character.
7. **`clear`** one last time.

## The 90-second beat sheet

Six beats. Each beat is one command. Type at a moderate pace — viewers
read with you. Pause 1-2 seconds after each command finishes before
running the next.

### Beat 1 — "What's installed?" (00:00 → 00:15)

```bash
mrm doctor
```

Shows two tables: drift detectors (KS / Wasserstein / Page-Hinkley /
MMD with their backends) and evidence signers (local / gpg / age /
kms / cloud-hsm paid-tier). Lets the viewer see the surface in 8
seconds without anything happening.

**Pause 2 seconds on the output.** The visual size of the two tables
already does the talking.

### Beat 2 — "What does a captured decision look like?" (00:15 → 00:30)

```bash
mrm replay reconstruct $(r)
```

Pretty-prints one full `DecisionRecord`: input_state, model_identity
(version + URI + framework), inference_params (seed / temperature),
output, content_hash, prior_record_hash.

**Pause 2 seconds.** A viewer needs time to see the four anatomical
pieces of replay (input / identity / params / output) on screen.

### Beat 3 — "Show the regulator-portal-export shape" (00:30 → 00:45)

```bash
mrm replay sample --model ccr_monte_carlo --n 5
```

Prints a `Replay sample (n=5)` table — Record ID, Model, Version,
Timestamp, Content hash. This is the *regulator-portal export* shape:
the command a bank would run when a Fed examiner asks "show me 50
random decisions from the last 90 days." Rich table renders cleanly.

**Pause 2 seconds** so the viewer reads at least the first row and
takes in the column shape.

**Note: we omit `mrm replay verify` from the demo** because the CCR
pickled model lives in a sub-package the bare CLI entrypoint cannot
import without `sys.path` injection (a known small CLI gap to file as
a follow-up issue). The `replay sample` command is the better demo
moment anyway -- it shows the *regulator-shaped use case* rather than
the engineering correctness.

### Beat 4 — "Is the chain intact?" (00:45 → 00:60)

```bash
mrm evidence root verify \
  --date 2026-05-17 \
  --chain-dir demo-artefacts/chain \
  --roots-dir demo-artefacts/roots \
  --signer local \
  --key-path demo-artefacts/root.key
```

Output ends with two green lines: `Root OK for 2026-05-17` / `signature
verified (local)` / `rederived matches published root`.

If the multi-line command feels clumsy, you can pre-stage it as a
shell function. Add this to `demo-setup.sh`'s cheatsheet exports:

```bash
alias verify-root='mrm evidence root verify --date 2026-05-17 \
  --chain-dir demo-artefacts/chain --roots-dir demo-artefacts/roots \
  --signer local --key-path demo-artefacts/root.key'
```

Then beat 4 becomes one line:

```bash
verify-root
```

That's prettier on a screen recording. **Pause 2 seconds.**

### Beat 5 — "What does the regulator see?" (01:00 → 01:25)

```bash
glow demo-artefacts/reports/ccr_monte_carlo_sr26_2.md
```

Or, if `glow` isn't installed:

```bash
bat demo-artefacts/reports/ccr_monte_carlo_sr26_2.md
```

Or worst-case:

```bash
less demo-artefacts/reports/ccr_monte_carlo_sr26_2.md
```

(`glow` is the prettiest — markdown rendered with colour. Install:
`brew install glow`. `bat` is the second-best — syntax-highlighted
plain text.)

Scroll through the rendered report for ~15 seconds:

- 2s on the title block ("SR 26-2 Model Validation Report")
- 4s on the executive-summary table (validation tests passed, pass
  rate, status: APPROVED)
- 4s on the SR 26-2 AI Evidence Posture section (the table showing
  `replay_backend: local`, `evidence_backend: local+Merkle`, with
  CONFIGURED markers)
- 4s on the SR 26-2 Compliance Matrix (the long table with anchor
  annotations `replay:decision_record` and `evidence:hash_chained_packet`)
- 1s on the findings section ("APPROVED FOR USE")

Press `q` to quit the viewer.

### Beat 6 — "Done." (01:25 → 01:30)

```bash
exit
```

This ends the asciinema recording cleanly. Don't add a closing
flourish, don't `clear`, don't `echo "thanks"`. The last visual the
viewer sees should be the rendered SR 26-2 report — that's the
artefact you're selling.

## Total timing

| Beat | Command | On-screen | Cumulative |
|---|---|---|---|
| 1 | `mrm doctor` | ~10s | 0:10 |
| pause | | 2s | 0:12 |
| 2 | `mrm replay reconstruct $(r)` | ~10s | 0:22 |
| pause | | 2s | 0:24 |
| 3 | `mrm replay verify …` | ~10s | 0:34 |
| pause | | 3s | 0:37 |
| 4 | `verify-root` | ~10s | 0:47 |
| pause | | 2s | 0:49 |
| 5 | `glow …sr26_2.md` (scroll) | ~25s | 1:14 |
| 6 | `exit` | ~1s | 1:15 |

**Comes in at ~75-90 seconds** which is exactly the target.

## After recording

```bash
# Upload to asciinema.org. Public link, free.
asciinema upload ~/riskattest-demo.cast
```

You'll get a URL like `https://asciinema.org/a/XXXXXXXX`. The output
also gives an embed snippet:

```html
<a href="https://asciinema.org/a/XXXXXXXX" target="_blank">
  <img src="https://asciinema.org/a/XXXXXXXX.svg" />
</a>
```

Drop that snippet at the top of your README, just under the badges.
GitHub renders it inline. **This is the artefact that converts repo
visitors.**

## If the take is bad

Re-record. asciinema casts are tiny JSON; re-running is faster than
trying to edit one. Common failures:

- **Typo** → re-record. Don't try to clean it up — asciinema's editor
  is painful.
- **Long pause that didn't auto-compress** → pass
  `--idle-time-limit=1` next take.
- **Wrong terminal size mid-recording** → re-record at the right size
  from the start.

Budget 3-5 takes. Most asciinema casts you watch online are take 4 or
5 of someone's own work.

## Three things NOT to do

- **No voiceover.** asciinema is text-only. Don't try to dub later;
  the medium is the message.
- **No fast typing.** Viewers read along. Aim for 100 WPM, not 200.
- **No `clear` between beats.** Each beat builds on the previous
  one's visual context. Clearing wastes the viewer's bearings.
