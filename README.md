# EIC Benchmark Report Generator

The EIC Benchmark Report Generator is a tool designed to generate visualisations based on data from various agencies.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contribute](#contribute)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Introduction

With data from seven different agencies, the EIC Benchmark Report Generator provides a unified visualization platform. This ensures standardized reporting and quick insights from the raw data.

## Features

- **Data Preprocessing**: Converts raw CSV data into a structured format for analysis.
- **Histograms**: Generates histograms for each category based on summed agency scores.
- **Radar Plots**: Showcases multi-dimensional data for agencies in a spider web representation.
- **Colored Tables**: Showcases tabular data with cells colored based on values.
- **Summarized Reports**: Aggregates data across agencies and provides a summative view.

## Installation

```bash
# Clone the repository
git clone https://github.com/innovation-growth-lab/eic-experimentation.git

# Navigate into the directory
cd eic_benchmark_report_generator

# If you have a virtual environment, activate it. Otherwise, proceed:
pip install -r requirements.txt
```

## Usage
You can generate the benchmark reports using the following command:

```bash
python benchmark_report.py --highlight [AGENCY_NAME] --data_path [DATA_PATH] --disclose --all
```

Arguments:
- `--highlight`: (Optional) Agency to highlight in the plots. Replace [AGENCY_NAME] with the desired agency's name.
- `--data_path`: (Optional) Path to the data file. By default, it will use the benchmarking_results.txt in the project's data directory.
- `--disclose`: (Optional) Whether to disclose the agency name in the plots. Without this flag, only the highlighted agency's name will be shown if provided.
- `--all`: (Optional) Iterate over each agency, highlighting it. If this option is used, it will generate reports for each agency separately.

For example, to generate a report highlighting the "EIC" agency without disclosing other agency names:

```bash
python benchmark_report.py --highlight EIC
```

To generate reports for each agency separately:

```bash
python benchmark_report.py --all
```

## License

This project is licensed under the MIT License.