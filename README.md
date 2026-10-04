# ECE 661 — Homework 5

## How to run

```bash
pip install -r requirements.txt
git clone https://github.com/magicleap/SuperGluePretrainedNetwork.git
cp superglue_ece661/{superglue_ece661.py,run_ece661.sh} SuperGluePretrainedNetwork/
python hw5_images/hw5_images/hw5_AdityaAbhang.py
```

All image paths are relative to the script (`DATA_DIR`, top of file). The SuperGlue repo is expected at
`../../SuperGluePretrainedNetwork` relative to the script, or set `SUPERGLUE_REPO=/path/to/repo`.
Figures and `run_log.txt` are written to `output/` next to the script. Task 5(b) runs once
`own_1.jpg` and `own_2.jpg` are placed next to the script.

## Seeds
The only randomness is the line colours and line subsets of the correspondence figures: seed 661 (`COLOR_SEED`).
Harris, NCC/SSD, SIFT and SuperPoint/SuperGlue inference are deterministic.

## Compute
CPU only (16 cores, 18 GB RAM, Linux 6.8), Python 3.11.16. Full run: about 80 s.
