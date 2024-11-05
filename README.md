This project uses mTAN (Multi-Time Attention Networks) to perform classification on multivariate time series data with missing values and evaluates the similarity of representation vectors between complete and partially missing datasets. 

## Overview
This code performs the following tasks:

1. **Dataset Preparation**: Loads a complete multivariate time series dataset, ensuring no missing values (PhysioNet or similar). 
2. **Missing Data Simulation**: Introduces missing values at varying proportions (10%, 20%, etc.) in the dataset to test the effect on model representations.
3. **Representation Extraction with mTAN**: Generates representation vectors for both the complete and missing data.
4. **Similarity Computation**: Calculates similarity metrics (Cosine Similarity, Mean Squared Error, Euclidean Distance) between representation vectors of the complete dataset and datasets with missing values.
5. **Result Reporting**: Summarizes similarity results at each missing data proportion.


## Installation

1. Clone the repository:

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   (Ensure you have PyTorch, NumPy, Scikit-learn, and Matplotlib installed.)


## Usage

To run the analysis, execute the following command:
```bash
python3 tan_classification.py --alpha 100 --niters 1 --lr 0.0001 --batch-size 50 --rec-hidden 256 --gen-hidden 50 --latent-dim 20 --enc mtan_rnn --dec mtan_rnn --n 8000 --quantization 0.016 --save 1 --classif --norm --kl --learn-emb --k-iwae 1 --dataset physionet
```

## Methodology

1. **Obtain Complete Data**: Loads a fully observed time series dataset without missing values.
2. **Introduce Missingness**: Adds random missing values at proportions like 10%, 20%, etc.
3. **mTAN Encoding**: Applies the mTAN model to obtain representation vectors.
4. **Similarity Analysis**: Compares the complete and incomplete representations using metrics like cosine similarity and MSE.
5. **Result Reporting**: Outputs similarity statistics for each level of missing data.



## Results

Results are printed to the console, showing similarity metrics (Cosine Similarity, MSE, Euclidean Distance) for each missing data level. For example:
```
Cosine Similarity at 10% Missing: Mean=0.89, Std=0.01
MSE at 10% Missing: Mean=0.05, Std=0.01
Euclidean Distance at 10% Missing: Mean=0.2, Std=0.05
...
```

