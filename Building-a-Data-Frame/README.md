# 📊 Java Data Analytics & Sampling Toolkit

This toolkit provides utilities for basic statistical analysis and data sampling in Java using Apache Commons Math.

## Features

- 📈 Statistical operations: t-tests, Pearson correlation, descriptive stats, linear regression.
- 📊 Tabular numeric data structure: filter, extend, and compute on rows/columns.
- 🔄 Sample synthetic data using Uniform, Gaussian, and Exponential distributions.

## Quick Example

```java
NumericTable table = new NumericTable(
    List.of("x", "y"),
    new double[][] {{1, 2}, {2, 4}, {3, 6}}
);

DataAnalytics analytics = new DataAnalytics(table);
System.out.println(analytics.pearsonsCorrelation("x", "y"));  // Output: ~1.0
```

## Requirements

- Java 8+
- Apache Commons Math 3.x

## Setup Guide

Follow these steps to get started with the Java Data Analytics Toolkit:

1. **Install Java 8 or higher**  
   Ensure you have Java 8+ installed. You can check by running:  
   ```bash
   java -version

**Disclaimer: Although I have extensive programming experience in Java, which I gained through bi-weekly practical projects in 'Introduction to Programming' and 'Programming' which focused on object-orientated programming, Most of these projects I cannot find in memory anymore.**

