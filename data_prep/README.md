# Data preparation

The configs read everything below `data/` in the repo (paths in a config are relative to the repo; an absolute path also works).

```
data/cc359/12ch/{train,val}/*.h5     data/cc359/index_12ch.json     data/cc359/masks/R{5,10}_218x{170,174,180}.npy
data/knee_pilot/multicoil_{train,val}/*.h5
data/cmrxrecon2023/h5/{train,val}/*.h5
```

## CC359 (Calgary-Campinas, 12-channel brain)

Download the Calgary-Campinas raw data (Multi-channel, 12-channel: `train_val_12_channel.zip`) and the challenge's sampling masks. Then:

```bash
python data_prep/prep_cc359.py --zip <download>/Raw-data/Multi-channel/12-channel/train_val_12_channel.zip --dst data/cc359/12ch --index data/cc359/index_12ch.json
```

and copy the challenge's Poisson-disc mask banks `R5_218x{170,174,180}.npy` and `R10_218x{170,174,180}.npy` (from its
`poisson_sampling` folder) to `data/cc359/masks/`. The split is PromptMR+'s, committed as `configs/splits/cc359_promptmr.json`: the
challenge's 47 training volumes for training, and PromptMR+'s 10 / 10 cut of the challenge's 20 validation volumes for validation
during training and for the test evaluation, with its fixed test mask per volume.

## fastMRI knee

Download fastMRI's knee `multicoil_val` set (199 volumes). Training and evaluation use 159 / 40 of these volumes:

```bash
python data_prep/make_knee_pilot.py --src <fastMRI>/knee/multicoil_val --dst data/knee_pilot
```

(symlinks only, split in `configs/splits/knee_pilot.json`).

## CMRxRecon2023 (cine / mapping)

Download the MICCAI 2023 challenge data (MultiCoil, `TrainingSet/FullSample`) and convert it with PromptMR+'s own script
([hellopipu/PromptMR-plus](https://github.com/hellopipu/PromptMR-plus), commit 934eeda, unchanged), which also applies their split
(P001-P100 train = 386 files, P101-P120 val = 76 files):

```bash
cd PromptMR-plus && python prepare_h5_dataset_cmrxrecon.py --input_matlab_folder <download>/ChallengeData/MultiCoil --output_h5_folder <repo>/data/cmrxrecon2023/h5/h5_dataset --split_json configs/data_split/cmr23-cardiac.json --year 2023
```

The script writes `h5_dataset/` plus `train/` and `val/` (symlinks); the config reads `data/cmrxrecon2023/h5/{train,val}`.
