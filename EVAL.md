# Evaluation — video-native goal detection

Hand-verified by watching the footage (burst frames per detected goal) and
building an objective ground truth from the on-screen scorebug.

## Ground truth (the scorebug is authoritative)

OCR of the scorebug every 20 s across the clip:

```
00:00–05:00  0-0
05:20        1-0   <-- goal 1 (scored ~05:00–05:20)
06:40        2-0   <-- goal 2 (scored ~06:20–06:40)
06:40–09:40  2-0   (final)
```

So the clip contains **exactly 2 goals**, final score **2-0**.
(The commentary narrates a 3-0 win; the footage looks AI-generated and the
commentary does not match the on-screen score, so the scorebug is used as truth.)

## Result: goal detection

The pipeline detected 7 goals. Matched against the 2 real goals (±20 s):

| detected | scorebug there | verdict |
|----------|----------------|---------|
| 03:34 | 0-0 | false positive |
| 04:18 | 0-0 | false positive |
| 05:05 | → 1-0 | TRUE (goal 1) |
| 05:29 | 1-0 | false positive (dup of goal 1) |
| 06:20 | → 2-0 | TRUE (goal 2) |
| 08:32 | 2-0 | false positive |
| 09:07 | 2-0 | false positive |

- True positives: **2** · False positives: **5** · False negatives: **0**
- **Recall = 2/2 = 100%** — it caught both real goals.
- **Precision = 2/7 ≈ 29%** — **false-goal rate ≈ 71%**.

## Interpretation

- The video model **finds every real goal** but **over-fires badly**: it labels
  shots, saves, and goalmouth scrambles as goals. Without a scoreboard, goal
  precision is ~29%.
- This quantifies the known trade: trusting the eyes gives full recall but low
  precision. The scorebug would filter the false positives — and in the app the
  optional "scoreboard ✓" flag does exactly that: it lights up on the 2 true
  goals only.
- The rich per-event detail also exposes the errors: several false "goals" have
  detail sentences that describe a save or a cross, not a ball crossing the line.

## What would raise precision (without depending on a scoreboard)

1. Ball-trajectory tracking — the only objective "ball crossed the line" signal.
2. A second verification pass per candidate goal ("does the ball actually enter
   the net in these frames? yes/no"), which our adversarial checks suggest helps.
3. Require net-ripple / ball-behind-line evidence, not just a shot toward goal.

## Method note

Repeatable for other event types: build a labelled window, run the same
match-within-window scoring. Goals were prioritised per the false-goal-rate ask.
