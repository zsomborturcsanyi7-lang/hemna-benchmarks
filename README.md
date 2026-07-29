# hemna-benchmarks

Benchmark suite and performance experiments for the HEMNA model architecture.

## Overview & Purpose
hemna-benchmarks provides evaluation suites for analyzing memory efficiency, inference latency, and perplexity across different configurations of the HEMNA architecture.

## Key Features
- Automated latency and memory profiling scripts.
- Perplexity computation on standard validation sets.
- Comparison utilities against baseline Transformer models.

## Tech Stack & Dependencies
- **Language**: Python 3.9+
- **Framework**: PyTorch

## Project Structure
```text
hemna-benchmarks/
├── benchmark_run.py
├── metrics.py
└── README.md
```

## Installation & Setup

### Prerequisites
- Python 3.9+

### Steps
```bash
git clone https://github.com/zsomborturcsanyi7-lang/hemna-benchmarks.git
cd hemna-benchmarks
pip install torch numpy
```

## Usage Examples
```bash
python benchmark_run.py
```

## Status & License
Status: Benchmark Suite.
License: MIT
