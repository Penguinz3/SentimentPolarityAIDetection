# Paper Notes

## Working Title

Multi-Signal AI Detector Reliability Analysis

## Research Question

Which interpretable text features are associated with false positives in probabilistic AI-text detection?

## Current Interpretation

The project began with sentiment polarity. The current framing is that sentiment appears to have statistically significant but practically weak explanatory power, so the next analysis layer tests additional linguistic and statistical signals. No claim is made that PCA or any single model proves separability between detector-positive and detector-negative human text.

## Analysis Guardrails

- Treat detector-positive labels on pre-2020 human-written text as false positives under the corpus assumptions.
- Keep detector scores and target labels out of feature matrices.
- Report feature importances as associations, not causal mechanisms.
- Prefer interpretable features over a larger but less focused technology stack.

