Overview
========

``litcoder`` is a modular Python library for building brain encoding models from naturalistic stimuli (text and speech). It emphasizes clean abstractions and composability:

- Assemblies: load and structure datasets (narratives, LPP, Lebel styles)
- Feature extractors: text and speech encoders with caching and multi-layer support
- Downsampling: align features to fMRI TRs
- FIR expansion: build time-lagged feature spaces
- Models: linear ridge, nested CV, and scikit-learn wrappers
- Training: a dependency-injected `AbstractTrainer` orchestrates the pipeline
- Logging & plotting: WandB/TensorBoard, brain plots, and run management

Design principles
-----------------

- Dependency injection for testability and reuse
- Clear data contracts between components
- Deterministic, disk-based caching for expensive features
- Separation of concerns (I/O, processing, modeling, orchestration) 