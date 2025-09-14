# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This repository contains research code for paper MK2740US: "Strategic Pricing for Multi-Platform Digital Publishing: A Penalty-Reward Optimization Approach" by Mustafif Khan. The project develops pricing optimization models for determining optimal pricing strategies across multiple digital publishing platforms while balancing revenue maximization with affordability constraints.

## Common Development Commands

### Running Optimization Models
```bash
# Run the main realistic economic model (recommended)
python code/realistic_economic_model.py

# Run diagnostic analysis to understand optimization failures
python code/diagnostic_analysis.py

# Run simplified solutions approach
python code/solutions.py

# Run basic logit model
python code/logit.py
```

### Data Collection and Analysis
```bash
# Generate dataset from Google Books API
python code/dataset.py

# Create visualizations and plots
python code/plot.py
python code/plots.py
python code/focused_visualization.py
```

### Testing Individual Components
```bash
# Test specific optimization approaches
python code/fixed_logit_model.py
python code/properly_fixed_logit_model.py
python code/true_beta_impact.py

# Run constraint analysis
python code/constraint_analysis.py
```

### Document Generation
```bash
# Compile main manuscript (requires Typst)
typst compile manuscript.typ

# Compile abstract
typst compile abstract.typ

# Compile conclusion
typst compile conclusion.typ
```

## Code Architecture

### Core Optimization Framework

The repository implements several approaches to the multi-platform pricing optimization problem:

1. **Realistic Economic Model** (`realistic_economic_model.py`) - The most sophisticated approach that incorporates:
   - Consumer demand curves with price elasticity
   - Platform operational costs 
   - Content quality multipliers based on royalty rates
   - Market size dynamics
   - Multiple beta scenarios for different market conditions

2. **Logit-Based Models** (`logit.py`, `logit_improved.py`, `fixed_logit_model.py`) - Market share allocation using logit functions with:
   - Utility functions incorporating royalty rates, market shares, and prices
   - Beta coefficients for price sensitivity, royalty preference, and market share effects
   - Constraint handling for price ordering and minimum gaps

3. **Diagnostic Framework** (`diagnostic_analysis.py`) - Analysis tools for understanding optimization failures:
   - Market size function behavior analysis
   - Revenue surface analysis
   - Gradient computation and sensitivity testing
   - Boundary behavior analysis

### Data Pipeline

- **Data Collection** (`dataset.py`) - Google Books API integration for programming book market data
- **Market Analysis** - Price vs page count relationships, statistical summaries
- **Visualization** (`plot.py`, `plots.py`, `focused_visualization.py`) - Comprehensive plotting utilities

### Mathematical Framework

The optimization problem is formulated as:
- **Objective**: Maximize expected revenue across platforms while penalizing excessive pricing
- **Variables**: Normalized prices `x_i ∈ [0,1]` mapped to actual prices `p_i`
- **Constraints**: Price monotonicity, royalty-based ordering, minimum gaps
- **Methods**: SLSQP (Sequential Least Squares Programming), Trust-Region Constrained

### Key Parameters

- Price bounds: `p_min = $8.99`, `p_max = $49.99` (or $50)
- Platform royalty rates: Range from 0.35 (Amazon Kindle) to 0.925 (own store)
- Market shares: Based on 2022 Canadian ebook market data
- Beta scenarios: Different consumer preference profiles (creator-focused, price-conscious, brand-loyal, etc.)

### Platform Configuration

The models work with 6 platform formats representing different digital publishing channels:
1. Format 1: Own store (92.5% royalty, ~11% market share)
2. Format 2: High-royalty platform (80% royalty, ~11% market share) 
3. Formats 3-5: Standard platforms (70% royalty, varying market shares)
4. Format 6: Major retailer (35% royalty, ~29% market share)

## File Organization

- `code/` - All Python optimization and analysis scripts
- `data/` - CSV datasets and processed data files
- `figures/` - Generated plots, visualizations, and result images
- `*.typ` - Typst source files for academic manuscript
- `*.pdf` - Compiled documents (manuscript, abstract, conclusion)
- `refs.bib` - Bibliography references

## Dependencies

Based on import analysis, the code requires:
```bash
pip install numpy scipy pandas matplotlib cvxpy requests
```

Additional requirements for document compilation:
- Typst (for .typ files)
- LaTeX packages for mathematical notation

## Development Notes

### Model Evolution

The repository shows iterative development of increasingly sophisticated models:
1. Basic logit models → Fixed logit models → Realistic economic models
2. Simple constraints → Complex economic forces → Interior equilibrium solutions
3. Single optimization attempts → Multiple starting points → Genetic algorithms

### Known Issues

From `diagnostic_analysis.py`, key challenges identified:
- Market size elasticity creating overwhelming incentives for low prices
- Boundary solutions instead of interior equilibria
- Need for balancing market size effects with price premiums

### Successful Approaches

The `realistic_economic_model.py` represents the most successful implementation:
- Achieves interior equilibria (non-boundary solutions)
- Incorporates realistic economic forces (demand curves, platform costs)
- Generates meaningful differentiation between beta scenarios
- Produces economically rational pricing strategies

## Research Context

This work is part of the MoKa Reads Collective's mission to "spend the least to support the most" - balancing fair author compensation with reader accessibility. The optimization framework supports data-driven pricing decisions for not-for-profit digital publishing.