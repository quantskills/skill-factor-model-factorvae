# Paper Mapping

This implementation follows the prior-posterior architecture of *FactorVAE: A
Probabilistic Dynamic Factor Model Based on Variational Autoencoder for
Predicting Cross-Sectional Stock Returns* (AAAI 2022): a posterior factor
distribution is learned from labels during training; a prior distribution is
predicted from historical characteristics; an alpha-beta decoder emits return
mean and uncertainty.

The repository is an independent implementation intended for arbitrary
user-supplied factor panels. Feature sets, labels, market universe, dates,
network widths, and portfolio construction are configured by the user and are
not expected to reproduce the paper's published numbers.

Paper: https://ojs.aaai.org/index.php/AAAI/article/view/20369
