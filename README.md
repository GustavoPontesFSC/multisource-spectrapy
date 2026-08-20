# Multisource-SpectraPy

Multisource-SpectraPy is a Python library under active development for
spectroscopic data analysis, preprocessing, chemometrics, machine learning,
and multi-source data fusion.

This repository contains the initial architecture of the fourth generation of
Data-Spectra. The project is being rebuilt around a clearer and more general
data model instead of extending the collection of specialized scripts present
in earlier versions.

> [!WARNING]
> This is an early development version. The available functionality is still
> limited, the public API may change without notice, and the package is not yet
> intended for production use.

## Design direction

The library distinguishes explicitly between a single spectrum and a spectral
dataset:

- `Spectrum` represents one pair of one-dimensional `x` and `y` axes.
- `Spectra` represents several spectra sharing an `x` axis, with each spectrum
  stored as a column of `Y` (`Y[:, i]`).

Low-level mathematical operations are kept independent from these data
objects whenever possible. `Spectrum` and `Spectra` will provide convenient
interfaces to those operations without making the numerical implementations
dependent on either class. This separation is intended to make the library
reusable, testable, and extensible to different spectroscopic techniques.

## Current state

The initial codebase currently establishes:

- the `Spectrum` and `Spectra` data models;
- shared axis-selection and baseline utilities;
- area-normalized Gaussian, Lorentzian, Voigt, and pseudo-Voigt profiles;
- the Schmid asymmetric pseudo-Voigt profile;
- the basic package and module organization.

These components are foundations for subsequent development, not a complete
replacement for the current Data-Spectra feature set.

## Development roadmap

The objective is to recover and expand the capabilities of Data-Spectra while
organizing them as composable numerical functions and stable object APIs.
The planned implementation order is:

1. **Foundation and reliability**
   - consolidate the invariants of `Spectrum` and `Spectra`;
   - define consistent mutation, copying, return-value, and metadata behavior;
   - add unit tests, typing, documentation, and continuous integration;
   - stabilize packaging and dependency declarations.

2. **Selection and preprocessing**
   - complete interval selection and exclusion operations;
   - implement baseline estimation and removal, including linear regression
     and asymmetric least squares;
   - add smoothing, normalization, derivatives, interpolation, resampling, and
     outlier-handling operations;
   - ensure that operations work consistently for one spectrum and for all
     columns of a spectral dataset.

3. **Peak analysis and fitting**
   - expand the library of symmetric and asymmetric spectral profiles;
   - implement peak detection and physically meaningful initial estimates;
   - support single-peak and multi-peak fitting;
   - expose fitted parameters, uncertainties, residuals, and fit-quality
     metrics in structured results.

4. **Dataset organization**
   - introduce spectrum identifiers, labels, units, acquisition information,
     and sample metadata;
   - support datasets whose spectra have individual `x` axes;
   - provide import and export interfaces without coupling analysis code to a
     particular file format;
   - define safe conversion paths to NumPy and tabular data structures.

5. **Chemometrics and machine learning**
   - implement dataset splitting, scaling, feature selection, and validation;
   - integrate PCA, PLS, classification, and regression workflows;
   - preserve preprocessing and model provenance;
   - expose spectral and fitted-peak features for downstream models.

6. **Multi-source data fusion**
   - align samples and metadata across analytical techniques;
   - support low-, intermediate-, and high-level fusion strategies;
   - combine raw spectra, selected regions, extracted peak parameters, and
     latent variables through consistent dataset interfaces;
   - provide workflows for evaluating whether fusion improves prediction and
     interpretation.

7. **Migration and stabilization**
   - compare the new implementations against representative Data-Spectra
     workflows;
   - document migration paths from earlier versions;
   - deprecate provisional interfaces gradually;
   - define the first stable public API after adequate test coverage and real
     experimental validation.

The roadmap describes direction rather than a fixed release schedule. Features
will be introduced incrementally, with tests and documentation accompanying
each stable implementation.

## Project status

Multisource-SpectraPy is currently a research-oriented work in progress.
Contributions and technical discussion will become more practical as the core
interfaces stabilize.

## License

This project is distributed under the BSD 3-Clause License. See `LICENSE` for
details.
