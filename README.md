# Something's Fishy in the Data Lake  
_Are Table Union Search Benchmarks Measuring Semantic Understanding?_  

This repository contains the code to reproduce the baseline results (Hash, TFIDF, Count Vectorizers, and SBERT) presented in our study. Please note that this release is for our baselines only. For replication of other methods such as STARMIE, HEARTS, or TabSketchFM, refer to their respective repositories:  

- **STARMIE:** [megagonlabs/starmie](https://github.com/megagonlabs/starmie)  
- **HEARTS:** [Allaa-boutaleb/HEARTS](https://github.com/Allaa-boutaleb/HEARTS)  
- **TabSketchFM:** [IBM/tabsketchfm](https://github.com/IBM/tabsketchfm)  

---

<p align="center">
    <a href="#getting-started"> Getting Started</a> • 
</p>

<p align="center">
    <a href="#datasets">Datasets</a> • 
    <a href="#license">License</a> •
    <a href="#citation">Citation</a>
</p>

---

## Overview

This repository provides a unified and clean implementation for evaluating Table Union Search benchmarks. The code is designed to work with a specific directory structure for the benchmarks and to quickly reproduce our experimental results on a variety of datasets. We have repackaged existing benchmarks to facilitate easier usage, but each benchmark requires some form of preprocessing to be fully compatible with our implementation.

---

## Getting Started

Follow these instructions to set up your environment, prepare the datasets, and run the experiments.

### 1. Create the Environment

First, create and activate the Conda environment with the required Python version:

```bash
conda create -n tus_benchmarks python=3.12 -y
conda activate tus_benchmarks
```

Then, install the necessary dependencies:

```bash
pip install -r requirements.txt
```

### 2. Prepare the Benchmarks

Our code assumes the following structure for any benchmark named `benchmark_name`:

```
data/
└── benchmark_name/
    ├── datalake/      # Contains CSV tables from the data lake
    ├── query/         # Contains CSV query tables
    └── benchmark.pkl  # Ground truth mapping
```

The `benchmark.pkl` file is a Python dictionary where:
- **Key:** name of the query table (e.g., `query.csv`)
- **Value:** list of unionable candidate table names (e.g., `[ "candidate_1.csv", "candidate_2.csv", ... ]`)

**Note:**
- The TUS and TUS-LARGE benchmarks already include the sampled queries used in our experiments (located under `query/`).
- The LakeBench benchmarks (LB-OPENDATA/LB-WEBTABLE) have undergone additional preprocessing, which includes:
  - Removal of tables not referenced in the ground truth.
  - Removal of ground truth entries missing a corresponding CSV file.
  - Adding the query table as a candidate to itself to standardize the unionable candidates.
- For LB-OpenData, a smaller version is provided where each table is limited to 1K rows for computational efficiency.

To quickly get started, download and extract the benchmarks into the `data/` folder using the provided link below:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15224451.svg)](https://doi.org/10.5281/zenodo.15224451)

Alternatively, you can download the original benchmarks from:

- **SANTOS:** [northeastern-datalab/santos](https://github.com/northeastern-datalab/santos)
- **TUS-SMALL/TUS-LARGE:** [RJMillerLab/table-union-search-benchmark](https://github.com/RJMillerLab/table-union-search-benchmark)
- **PYLON:** [superctj/pylon](https://github.com/superctj/pylon/tree/main)
- **UGEN-V1/UGEN-V2:** [northeastern-datalab/gen](https://github.com/northeastern-datalab/gen)
- **LB-OPENDATA/LB-WEBTABLE:** [RLGen/LakeBench](https://github.com/RLGen/LakeBench)

### 3. Run the Experiments

To produce search results using a specified embedding approach on a given benchmark, run:

```bash
python run.py BENCHMARK_NAME [--methods METHOD1 METHOD2 ...] [--k K_VALUE] [--limit LIMIT] [--exclude-self-matches]
```

For example, to run experiments on the `santos` benchmark using Hash, TFIDF, Count Vectorizers, and SBERT with a top-10 candidate selection:

```bash
python run.py santos --methods hash tfidf count sbert --k 10
```

The code will:
- Generate embeddings for each method (saving files to the `vectors/` directory),
- Execute search queries (saving results to the `results/` directory).

---

## Datasets

The datasets used for benchmarking have been repackaged into a unified format to facilitate reproducibility. After extracting the benchmarks into the `data/` folder, each benchmark should adhere to the directory structure described above. Please refer to the individual benchmark documentation for any further preprocessing details.

---

## Profiling

For those interested in analyzing the profile of a given benchmark in terms of data distribution and the overlap in ground truth unionable pairs, use the scripts located in the [`profiler/`](./profiler/) directory. Refer to the README inside the `profiler/` folder for detailed instructions.

---

## License

This code is released under the [CC BY-NC-ND 4.0 License](LICENSE).  
*You are free to copy, modify, and distribute this code for non-commercial purposes, subject to the terms of the license.*

---

## Citation

[1] A. Boutaleb, A. Almutawa, B. Amann, R. Angarita, and H. Naacke. HEARTS: Hypergraph-based Related Table Search. In ELLIS Workshop on Representation Learning and Generative Models for Structured Data, 2025.
[2] G. Fan, J. Wang, Y. Li, D. Zhang, and R. J. Miller. STARMIE: Semantics-aware dataset discovery from data lakes. In Proceedings of the VLDB Endowment (PVLDB), 16(7):1726–1739, 2023.
[3] Khatiwada, A., Kokel, H., Abdelaziz, I., et al. (2025) - TabSketchFM: Sketch-based Tabular Representation Learning for Data Discovery over Data Lakes. IEEE ICDE.
[4] F. Nargesian, E. Zhu, K. Q. Pu, and R. J. Miller. TUS: Table union search on open data. In Proceedings of the VLDB Endowment (PVLDB), 11(7):813–825, 2018.
[5] A. Khatiwada, G. Fan, R. Shraga, Z. Chen, W. Gatterbauer, R. J. Miller, and M. Riedewald. SANTOS: Relationship-based semantic table union search. In Proceedings of the ACM on Management of Data (SIGMOD), 1(1):1–25, 2023.
[6] Y. Deng, C. Chai, L. Cao, Q. Yuan, S. Chen, Y. Yu, Z. Sun, J. Wang, J. Li, Z. Cao, K. Jin, C. Zhang, Y. Jiang, Y. Zhang, Y. Wang, Y. Yuan, G. Wang, and N. Tang. LakeBench: A Benchmark for Discovering Joinable and Unionable Tables in Data Lakes. In Proceedings of the VLDB Endowment, 17(8):1925–1938, 2024.
[7] K. Pal, A. Khatiwada, R. Shraga, and R. J. Miller. ALT-GEN: Benchmarking Table Union Search using Large Language Models. In Proceedings of the VLDB Endowment, 2150:8097, 2024.

---

## Acknowledgments

We acknowledge the original authors and maintainers of the benchmarks used in this repository, as well as the developers of the various baseline methods. Your contributions have been invaluable to our research.

