# Table of Contents

- [Layer Normalization](#layer-normalization)
  - [LayerNorm vs Generic Normalization](#layernorm-vs-generic-normalization)
- [BatchNorm](#batchnorm)
  - [What It Solves](#what-it-solves)
    - [Vanishing / Exploding Gradient](#vanishing--exploding-gradient)
    - [Internal Covariate Shift Problem](#internal-covariate-shift-problem)

# Layer Normalization
Layer normalization is a technique used in NN's to normalize the activations of a layer, making training more stable and faster

Basically, it helps to ensure activations (inputs) have "normal", or at least more consistent, distribution

It's the same idea as why we normalize our features for Regression, Classification, or any other model - it ensures consistency and generalization so that one feature / input doesn't dominate the entire model

Layer Normalization operates by calculating the mean and variance of the inputs for each sample, and then applying normalization transformation on top of it

## LayerNorm vs Generic Normalization
TLDR;
- For each sample (row), you compute the mean and variance across that sample’s features.
- Each feature in the row is normalized using the statistics (mean/variance) of that row only.
- No information from other samples is used.

The main difference in LayerNorm is our input - in LayerNorm we will usually get batches of data into our NN, and the batches are made up of multiple rows (samples)

LayerNorm normalizes all of the features in that sample using the mean and variance of that sample, ***it does NOT use information from other samples in the batch***

|      |      |      |
|------|------|------|
| a₁₁ | a₁₂ | a₁₃ |
| **a₂₁** | **a₂₂** | **a₂₃** |
| a₃₁ | a₃₂ | a₃₃ |

***This basically means we normalize over the row, and not the column*** which is the main difference from BatchNorm as well

For input vector $x$ of length $d$ - $(x_1, x_2, ..., x_d)$:
$\mu = \frac{1}{d} \sum_{i=1}^d x_i$ 

$ \sigma^2 = \frac{1}{d} \sum_{i=1}^d (x_i - \mu)^2 $

$\text{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}$


## BatchNorm
TLDR;
- For each feature (column), you compute the mean and variance across the batch (i.e., across all samples (rows) for that feature (column)).
- Each feature is normalized using the statistics of that feature across all samples in the batch.
- No information from other features in the same row is used.

BatchNorm is equivalent to LayerNorm, except our features are calculated across the feature across the entire batch, basically meaning ***we normalize over the column and not the row***

|      |  |      |
|------|------|------|
| a₁₁ | **a₁₂** | a₁₃ |
| a₂₁ | **a₂₂** | a₂₃ |
| a₃₁ | **a₃₂** | a₃₃ |


$\mu_j = \frac{1}{k} \sum_{i=1}^k x_{i,j}$

$\sigma_j^2 = \frac{1}{k} \sum_{i=1}^k (x_{i,j} - \mu_j)^2$

$\text{BatchNorm}(x_{i,j}) = \frac{x_{i,j} - \mu_j}{\sqrt{\sigma_j^2 + \epsilon}}$

# What It Solves

## Vanishing / Exploding Gradient

## Internal Covariate Shift Problem