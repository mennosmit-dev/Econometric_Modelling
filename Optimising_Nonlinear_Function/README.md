# Lagrangian Relaxation for the Set Covering Problem

MATLAB implementation of a Lagrangian relaxation framework for solving 
the set covering problem using subgradient optimization.

The project explores dual optimization techniques, heuristic feasibility recovery, 
and iterative bound improvement for combinatorial optimization problems.

---

## 🧠 Overview

Core components:

- Lagrangian relaxation of covering constraints
- Subgradient-based multiplier updates
- Greedy heuristic for primal feasibility
- Iterative bound improvement via dual optimization

The implementation demonstrates classical operations research techniques 
for large-scale combinatorial optimization.

---

## 🔍 Problem Setting

The set covering problem selects a subset of sets ("routes") that covers all 
elements ("customers") while minimizing total cost.

Lagrangian relaxation moves difficult constraints into the objective function 
using multipliers, enabling efficient approximation of lower bounds.

---

## ⚙️ Implemented Functions

### 🔹 `CalculateLagrangian`
```matlab
[obj_lagrange, x_lagrange] = CalculateLagrangian(c, A, lambda)
```
Computes the Lagrangian objective value and corresponding solution
for a given multiplier vector.

### 🔹 ComputeNextLambda

Updates Lagrangian multipliers using a subgradient step based on
constraint violations.

### 🔹 ConvertInfeasToFeas

Transforms infeasible solutions into feasible primal solutions
using a greedy repair heuristic.

### 🔹 PerformSubgradientOptimization

Runs the full subgradient optimization loop to iteratively improve
dual bounds and track convergence behavior.

## 🔧 Tech Stack

MATLAB • Operations Research • Combinatorial Optimization

## 📌 Context

This repository complements my broader work in optimization, econometrics,
and machine learning by demonstrating classical dual optimization techniques
and algorithmic design for structured decision problems.
```matlab
[obj_lagrange, x_lagrange] = CalculateLagrangian(c, A, lambda)
