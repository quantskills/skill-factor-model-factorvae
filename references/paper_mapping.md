# Paper Mapping

## Reference

Yitong Duan, Lei Wang, Qizhong Zhang, and Jian Li. *FactorVAE: A Probabilistic Dynamic Factor Model Based on Variational Autoencoder for Predicting Cross-Sectional Stock Returns.* AAAI 2022, pp. 4468-4476. DOI: https://doi.org/10.1609/aaai.v36i4.20369

## Reproduction status

This repository is an **unofficial independent reproduction and generic factor-panel adaptation**. It is not affiliated with or released by the paper authors.

## Preserved method structure

- historical sequence encoder;
- label-informed posterior latent-factor distribution during training;
- feature-informed prior latent-factor distribution for inference;
- dynamic cross-sectional portfolio aggregation;
- Gaussian alpha-beta decoder and posterior-prior KL objective.

## Configurable adaptation surface

The input factors, label horizon, time split, model width, latent-factor count,
portfolio count, and optimizer settings are supplied by the caller. This makes
the package suitable for generic factor panels but means it is not intended to
reproduce the paper's particular dataset or reported numerical results.
