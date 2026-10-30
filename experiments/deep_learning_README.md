# Deep learning implementation status

**Implemented source:** differentiable PyTorch angular-spectrum operator, compact baseline phase CNN, physics-unrolled phase optimizer, deterministic synthetic smoke-training entrypoint, optional tests.

**Not yet implemented/verified:** original HoloNet reproduction; dataset-specific loaders; complete experimental comparisons; quantized SLM and camera model; cross-platform training verification; trained production research weights; measured optical data.

The PhaseCNN is NOT HoloNet; the unrolled model is a prototype to be assessed against peer-reviewed methods. Toy training uses synthetic Gaussian targets solely to test numerical execution. No toy scores may be represented as published empirical results. Training should use scientifically defined optical targets with documented dataset sources and leakage-safe splits.
