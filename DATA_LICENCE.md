# Data licence and attribution

Kasauti's code is Apache-2.0. The data it reads is not ours and is not in this repository.

## Volve

The Volve field dataset was disclosed in 2018 by Equinor ASA and the former Volve licence
partners (ExxonMobil Exploration and Production Norway AS and Bayerngas Norge AS).
It is offered under the **Equinor Open Data Licence**, published as "Terms and conditions for
licence to data: Volve":

https://www.equinor.com/content/dam/statoil/documents/what-we-do/Equinor-HRS-Terms-and-conditions-for-licence-to-data-Volve.pdf

What we rely on, and where each point comes from:

| Point | Source |
|---|---|
| The licence states it is based on CC BY 4.0, with modifications | Quoted in Agile Scientific, 20 May 2021 |
| Credit must be given to Equinor and the former Volve licence partners, with a link to the terms | Same |
| The data must not be presented in a misleading or distorted way | Same |
| Selling the data as such is restricted; Equinor's stated position is that value must be added | Equinor e-mail quoted in the same article |
| Research, study and development use needs no separate permission | Equinor's Volve data-sharing page |

**The licence PDF itself could not be opened from the build environment for this release, so the
table above rests on secondary sources. Read the PDF before you publish, redistribute data, or
use Kasauti in paid work.** This file is not legal advice.

## What Kasauti does to stay inside these terms

1. No Volve file is committed. `data/` is git-ignored. `kasauti fetch` downloads five files to
   your machine and checks each SHA-256 against the manifest in `src/kasauti/data.py`.
2. Published results are aggregates (error rates, coverage, counts). No row-level Volve values
   are written to `results/`.
3. Every results file and figure caption carries the attribution line below.
4. Nothing here suggests that Equinor or its partners endorse Kasauti or its findings.

Attribution line to reuse:

> Contains data from the Volve field dataset, disclosed by Equinor ASA and the former Volve
> licence partners under the Equinor Open Data Licence.

## Where the five files come from

Equinor distributes Volve through its data portal (https://www.equinor.com/energy/volve-data-sharing),
which needs a login. For a one-command, checksum-verified set-up, `kasauti fetch` reads from a
public mirror pinned to one commit:

`github.com/yohanesnuwara/volve-machine-learning` at `3b089e0e50fc69207b84616c6cb364b5c5ca2960`

| File | Note |
|---|---|
| `Volve production data.xlsx` | Daily and monthly production workbook |
| `15_9-F-1A.LAS`, `15_9-F-1B.LAS`, `15_9-F-11A.LAS` | Composite well logs |
| `Volve_well_picks_modified.csv` | Formation picks. The file name says "modified": the mirror's author re-formatted Equinor's picks file. It is used only by two agent tools, not by Track S or Track P |

Two things follow. First, the mirror is a third party; the checksums prove you have the same bytes
we used, not that those bytes equal Equinor's originals. Second, if you obtain the files from the
official portal and a checksum differs, Kasauti will stop. That is intended: results from
different bytes are not comparable.
