# yatha

Source code for the undergraduate thesis project.

## Step 1: EEG Preprocessing

```bash
python main.py run=preprocess dataset=pumc_sev process=archaic-10
```

`MNE-Python` ([project](https://doi.org/10.5281/zenodo.592483), [paper](doi:10.3389/fnins.2013.00267)) and `autoreject` ([paper1](https://hal.archives-ouvertes.fr/hal-01313458/document), [paper2](http://www.sciencedirect.com/science/article/pii/S1053811917305013)) are used to build the whole preprocessing workflow and automatically remove bad epochs, respectively.

Procedure:

- Discard the first 20s
- Band-pass filter at 0.1-50Hz
- Notch filter at 50Hz
- A1-A2 re-reference
- Segment to 10s epochs
- Remove bad epochs via `autoreject`
- Downsample to 250Hz

## Step 2: Feature Extraction

```bash
python main.py run=extract dataset=pumc_sev process=archaic-10
```

`brainfeatures` ([github](https://github.com/TNTLFreiburg/brainfeatures), [paper](https://www.sciencedirect.com/science/article/pii/S1053811920305073)) is used to extract features from EEG signals. Five types of features are extracted:

1. Continuous wavelet transform (CWT)
2. Discrete wavelet transform (DWT)
3. Discrete Fourier transform (DFT)
4. Phase locking value
5. Time domain features

After feature extraction, each sample is transformed into a feature vector consisting of 5,700 feature values, which serves as input data for the subsequent model. The reason for using the `brainfeatures` library is its following advantages:

1. When computing features, a 10s sample signal is segmented into multiple 1s windows. The feature values are computed for each window, and the median is taken to remove outliers. 
2. Within each 1s window, abnormal signal values are further removed to enhance the stability of the computed features. Additionally, a window function is applied to reduce frequency leakage.
3. After computing the feature values, statistical information such as the maximum, minimum, mean, and variance of these coefficients is calculated, while the coefficients themselves are not retained. This approach further condenses the information in the feature values and eliminates a large amount of irrelevant data.

Pre-check: `python resplit.py --split sev check`

Resplit data:

1. `python resplit.py --split age resplit`
2. `python resplit.py --split sev resplit`

First resplit `age`, then `sev`, because there are 2 samples that should be present in `age` but not `sev`.

Check data availability:

1. `python resplit.py --split age check`
2. `python resplit.py --split sev check`

Obtain data statistics:

1. `python resplit.py --split age stat`
2. `python resplit.py --split sev stat`

## Step 3: Model Training

```bash
python main.py run=train dataset=pumc_sev process=archaic-10 dataset/task=sev_s+mo+mi-n model.dropout_p=0.1 train.device_ids=4
```

RRL ([paper1](https://proceedings.neurips.cc/paper/2021/hash/ffbd6cbb019a1413183c8d08f2929307-Abstract.html), [paper2](https://ieeexplore.ieee.org/abstract/document/10302393)) is selected for its excellent performance and outstanding interpretability.