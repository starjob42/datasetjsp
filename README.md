


# Cloning the repository and downloading the dataset

This repository contains large files managed by Git Large File Storage (LFS). Follow the steps below to ensure you can properly clone the repository and access the large files.

## Prerequisites

- Git
- Git LFS (download and install from [Git LFS website](https://git-lfs.github.com/))

## Setup Instructions

### Step 1: Install Git LFS

Ensure Git LFS is installed on your system. If it is not already installed, you can install it by running:

```sh
git lfs install

### Step 2: Clone the Repository

git clone https://github.com/starjob42/datasetjsp.git
cd datasetjsp

Step 3: Pull LFS Objects
After cloning the repository, ensure Git LFS pulls the large files:

sh
Copy code
git lfs pull






# Job Shop Scheduling Dataset Statistics

## General Statistics
- **total_samples**: 120000
- **unique_sizes**: 50
- **data_size_of_group_size_of_instances**: 2400
- **average_jobs**: 8.24
- **average_machines**: 5.64
- **average_makespan**: 1434.3538358961537
- **makespan_variance**: 1292087.1381092232
- **median_makespan**: 1211.0
- **min_makespan**: 5.0
- **max_makespan**: 9852.0

## Size Distribution
![Size Distribution](plots/size_distribution.png)

## Average Makespan per Size
![Average Makespan per Size](plots/average_makespan_per_size.png)

## Variance of Makespan per Size
![Variance of Makespan per Size](plots/variance_makespan_per_size.png)

## Median Makespan per Size
![Median Makespan per Size](plots/median_makespan_per_size.png)

## Minimum and Maximum Makespan per Size
![Min and Max Makespan per Size](plots/min_max_makespan_per_size.png)

## Correlations
- **correlation_jobs_makespan**: 0.5658973838191064
- **correlation_machines_makespan**: 0.5480905138716899

## Histograms
![Histogram of Jobs and Machines](plots/jobs_machines_histogram.png)

![Histogram of Makespan](plots/makespan_histogram.png)
