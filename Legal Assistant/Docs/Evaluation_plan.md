
# Evaluation Plan

This document describes how the system is evaluated.

## Legal Question Answering Evaluation
- Retrieval Recall@K is used to measure retrieval accuracy.
- The evaluation checks whether the correct legal article is retrieved.
- Both Arabic and English queries are tested.

## Contract Drafting Evaluation
- Rule-based validation is applied.
- The system checks for:
  - Employee name
  - Salary
  - Legal disclaimer
- These checks ensure basic correctness of the generated contract.
