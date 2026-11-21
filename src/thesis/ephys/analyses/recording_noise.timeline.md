# Recording noise survey

Question: which recordings are noisy, which kind of noise it is, and does it
follow the probe reference or the implant age?

Entries before 2026-11-21 are reconstructed from the git log, the module
docstring, saved result files and agent notes. See
`ephys/analyses/glm/timeline.md` for the conventions.

## 2026-11-18 · finding · AI
What: 60 recordings, 11 mice. GRB068 (gnd reference): 8–30 % of 1 s windows
saturate with a 60 Hz common mode larger than the ADC range. Joined-tips mice
have 0 % saturation. GRB064 floor is flat over 68 days. Reference is
confounded with mouse.
Evidence: de805aa3; Notion rig audit https://app.notion.com/p/40aecf086b7c819aa98cee4d2ab1502b

## 2026-11-19 · finding · AI
What: GRB068 saturation is locked to the side ports (lick artifact): after a
rewarded entry the rail fraction goes from 1 % to 21–25 %.
Evidence: Notion rig audit https://app.notion.com/p/40aecf086b7c819aa98cee4d2ab1502b, "GRB068 saturation is locked to the side ports"

## 2026-11-19 · decision · Gabriel
What: Ground the water in the spout to the OBX breakout ground.
Why: It removed the lick saturation, seen live.

## 2026-11-19 · correction · AI
What: Corrects the 2026-11-19 decision. With joined tips and the water
ground, GRB061 goes to the ADC limit after 99 % of rewarded trials (0 % with
tip reference and no water ground). GRB068 improves (28 → 11 %).
Evidence: Notion rig audit https://app.notion.com/p/40aecf086b7c819aa98cee4d2ab1502b, "GRB061 survey with joined tips"

## 2026-11-20 · finding · AI
What: Joined tips with no water ground: the lick artifact is gone. 6 ADC
channels are stuck.
Evidence: Notion "Lick-noise test recordings"
https://app.notion.com/p/41decf086b7c817192cac094a623609a
