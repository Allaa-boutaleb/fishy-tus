# 🐟 Something's Fishy in the Data Lake  
_Are Table Union Search Benchmarks Measuring Semantic Understanding?_  

This repository contains the code to reproduce the baseline results (Hash, TFIDF, Count Vectorizers, and SBERT) presented in our study. Please note that this release includes **only our baseline implementations**. For other methods like STARMIE, HEARTS, or TabSketchFM, please refer to their respective repositories:

- **STARMIE:** [megagonlabs/starmie](https://github.com/megagonlabs/starmie)  
- **HEARTS:** [Allaa-boutaleb/HEARTS](https://github.com/Allaa-boutaleb/HEARTS)  
- **TabSketchFM:** [IBM/tabsketchfm](https://github.com/IBM/tabsketchfm)  

---

<p align="center">
    <a href="#rocket-getting-started">🚀 Getting Started</a> • 
    <a href="#open_file_folder-datasets">📂 Datasets</a> • 
    <a href="#bar_chart-profiling">📊 Profiling</a> • 
    <a href="#scroll-license">📜 License</a> • 
    <a href="#bookmark-citation">🔖 Citation</a>
</p>

---

## 🔍 Overview

This repository provides a unified and clean implementation for evaluating Table Union Search benchmarks. The code supports a standardized structure across different datasets and enables reproducible experiments with minimal configuration.  

We have repackaged all datasets in a unified format for **ease of use**. However, **original benchmark data must be processed and formatted** according to our schema before use — including file restructuring, query sampling, and in some cases additional cleaning.

---

## 🚀 Getting Started

Follow these steps to set up your environment and prepare the benchmarks.

### 1. Set Up the Environment

Create and activate the environment:

```bash
conda create -n tus_benchmarks python=3.12 -y
conda activate tus_benchmarks
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

### 2. Download the Benchmarks

> [!IMPORTANT]
> 🧳 To quickly get started, download all the processed benchmarks from Zenodo: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15224451.svg)](https://doi.org/10.5281/zenodo.15224451). This archive includes all benchmarks preprocessed and formatted as required by this repository. All you have to do is extract them into the `data/` folder.

Alternatively, you can obtain the **original datasets** from the authors’ repositories. However, **you will need to preprocess them**:

- Convert them into our unified format (`datalake/`, `query/`, `benchmark.pkl`)
- Sample query tables for **TUS-SMALL** and **TUS-LARGE**
- Apply further cleaning and structure enforcement for **LakeBench OpenData** and **WebTable**

Original links of benchmarks:

- **SANTOS:** [northeastern-datalab/santos](https://github.com/northeastern-datalab/santos)
- **TUS-SMALL / TUS-LARGE:** [RJMillerLab/table-union-search-benchmark](https://github.com/RJMillerLab/table-union-search-benchmark)
- **PYLON:** [superctj/pylon](https://github.com/superctj/pylon/tree/main)
- **UGEN-V1 / UGEN-V2:** [northeastern-datalab/gen](https://github.com/northeastern-datalab/gen)
- **LB-OPENDATA / LB-WEBTABLE:** [RLGen/LakeBench](https://github.com/RLGen/LakeBench)

Expected structure:

```
data/
└── benchmark_name/
    ├── datalake/      # CSV tables from the data lake
    ├── query/         # CSV query tables
    └── benchmark.pkl  # Python dict {query_name: [candidate_1, candidate_2, ...]}
```

📝 Notes:
- `TUS-SMALL` and `TUS-LARGE` include the **sampled query subsets** used in our experiments, located under the `query/` directory. These were selected from the full benchmark to match our experimental setup.  
  ➤ You can still access **all original queries** from both benchmarks in the `all_query/` folder available in the archives listed in [here](https://doi.org/10.5281/zenodo.15224451).
  
- `LB-OPENDATA` and `LB-WEBTABLE` have undergone **extensive preprocessing** to ensure consistency and usability:
  - Tables from the data lake that are **not referenced** in the ground truth were removed.
  - Ground truth entries pointing to **missing candidate tables** were also filtered out.
  - For each query, the **query table itself was added as a candidate** to maintain consistency across benchmarks.
  - For `LB-OPENDATA`, we additionally provide a **reduced version** where each table is truncated to a **maximum of 1,000 rows**, allowing for more efficient experimentation.
### 3. Run Experiments

To run a benchmark using the selected methods:

```bash
python run.py BENCHMARK_NAME [--methods METHOD1 METHOD2 ...] [--k K_VALUE] [--limit LIMIT] [--exclude-self-matches]
```

Example:

```bash
python run.py santos --methods hash tfidf count sbert --k 10
```

The output will include:
- Vector files in `vectors/`
- Ranking results in `results/`

---

## 📂 Datasets

All datasets used in our study follow the same directory structure and format. If you use the provided Zenodo archive, everything will be ready to go. If using original datasets:

- Follow the format explained in [Getting Started](#getting-started)
- Consult original documentation for preprocessing steps

---

## 📊 Profiling

You can analyze the structure of any benchmark (e.g., query-candidate overlaps, attribute statistics) using the scripts in the [`profiler/`](./profiler/) folder. See the README inside for usage examples.

---

## 📜 License

This code is released under the [CC BY-NC-ND 4.0 License](LICENSE).  
*You may copy, use, and share this code for **non-commercial** purposes with attribution.*

---

## 🔖 Citation

Please cite our work and related benchmarks as follows:

[1] A. Boutaleb, A. Almutawa, B. Amann, R. Angarita, and H. Naacke. HEARTS: Hypergraph-based Related Table Search. In *ELLIS Workshop on Representation Learning and Generative Models for Structured Data*, 2025.  
[2] G. Fan, J. Wang, Y. Li, D. Zhang, and R. J. Miller. STARMIE: Semantics-aware dataset discovery from data lakes. In *Proceedings of the VLDB Endowment (PVLDB)*, 16(7):1726–1739, 2023.  
[3] A. Khatiwada, G. Fan, R. Shraga, Z. Chen, W. Gatterbauer, R. J. Miller, and M. Riedewald. SANTOS: Relationship-based semantic table union search. In *Proceedings of the ACM on Management of Data (SIGMOD)*, 1(1):1–25, 2023.  
[4] K. Pal, A. Khatiwada, R. Shraga, and R. J. Miller. ALT-GEN: Benchmarking Table Union Search using Large Language Models. In *Proceedings of the VLDB Endowment*, 2150:8097, 2024.  
[5] Y. Deng, C. Chai, L. Cao, Q. Yuan, S. Chen, Y. Yu, Z. Sun, J. Wang, J. Li, Z. Cao, K. Jin, C. Zhang, Y. Jiang, Y. Zhang, Y. Wang, Y. Yuan, G. Wang, and N. Tang. LakeBench: A Benchmark for Discovering Joinable and Unionable Tables in Data Lakes. In *Proceedings of the VLDB Endowment*, 17(8):1925–1938, 2024.  
[6] F. Nargesian, E. Zhu, K. Q. Pu, and R. J. Miller. TUS: Table union search on open data. In *Proceedings of the VLDB Endowment (PVLDB)*, 11(7):813–825, 2018.  
[7] A. Khatiwada, H. Kokel, I. Abdelaziz, et al. TabSketchFM: Sketch-based Tabular Representation Learning for Data Discovery over Data Lakes. In *IEEE ICDE*, 2025.

---

## 🙏 Acknowledgments

We thank the creators and maintainers of all benchmarks and baseline models that made this research possible.
