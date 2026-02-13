# Java Data Analytics & Sampling Toolkit

Lightweight Java utilities for statistical analysis, tabular data manipulation, 
and synthetic data generation built on top of Apache Commons Math.

This toolkit was developed to explore object-oriented implementations of 
statistical workflows and reusable data structures in Java.

---

## 🧠 Overview

Core capabilities:

- Statistical analysis:
  - t-tests
  - Pearson correlation
  - descriptive statistics
  - linear regression

- Tabular numeric data structure:
  - column-based filtering
  - row/column operations
  - extendable datasets

- Synthetic data generation:
  - Uniform distribution
  - Gaussian distribution
  - Exponential distribution

---

## 🚀 Example

```java
NumericTable table = new NumericTable(
    List.of("x", "y"),
    new double[][] {{1, 2}, {2, 4}, {3, 6}}
);

DataAnalytics analytics = new DataAnalytics(table);
System.out.println(analytics.pearsonsCorrelation("x", "y"));  // Output: ~1.0```

## Requirements

- Java 8+
- Apache Commons Math 3.x

## Setup Guide

Follow these steps to get started with the Java Data Analytics Toolkit:

1. **Install Java 8 or higher**  
   Ensure you have Java 8+ installed. You can check by running:  
   ```bash
   java -version

