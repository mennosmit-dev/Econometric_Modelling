# Lagrangian Relaxation for Set Covering Problem

This repository provides MATLAB functions to solve the **set covering problem** using **Lagrangian relaxation** and **subgradient optimization**.

---

## 📖 Overview

The code implements:

- **CalculateLagrangian:** Computes the Lagrangian relaxation’s objective and solution for given multipliers.
- **ComputeNextLambda:** Updates Lagrangian multipliers via a subgradient step.
- **ConvertInfeasToFeas:** Converts infeasible primal solutions into feasible ones with a greedy heuristic.
- **PerformSubgradientOptimization:** Runs the overall subgradient optimization algorithm to improve bounds iteratively.

---

## 🔍 Problem Description

The **set covering problem** aims to select a subset of "routes" (or sets) to cover all "customers" (or elements), minimizing total cost. Lagrangian relaxation allows handling difficult constraints by incorporating them into the objective with multipliers.

---

## ⚙️ Functions

### 1. `CalculateLagrangian`

```matlab
[obj_lagrange, x_lagrange] = CalculateLagrangian(c, A, lambda)
```


**Disclaimer: Although I have extensive programming experience in MATLAB, acquired through practical projects in combinatorial optimization and theoretical seminars (including operations research), many of these projects are no longer in my memory or available for reference.**

