# GitHub Copilot Repository Instructions

## Project Overview
This repository contains a series of data science and machine learning tasks implemented in Python. Each task is organized in its own folder (`Task 1` through `Task 9`), covering topics such as data cleaning, EDA, regression, classification, clustering, and more.

## Code Style & Conventions
- **Language**: Python 3.8+
- **Style guide**: PEP 8 (4-space indentation, 79-character line limit)
- **Docstrings**: Google-style docstrings for all public functions and classes
- **Imports**: Standard library first, then third-party, then local — each group separated by a blank line
- **Naming**: `snake_case` for variables and functions; `PascalCase` for classes; `UPPER_SNAKE_CASE` for constants
- **Type hints**: Add type annotations to all function signatures

## Libraries in Use
- `pandas`, `numpy` — data manipulation
- `matplotlib`, `seaborn` — visualisation
- `scikit-learn` — machine learning models and preprocessing
- `scipy` — statistical utilities

## Testing & Quality
- Keep functions small and single-purpose
- Validate inputs at function boundaries and raise meaningful exceptions
- Avoid magic numbers — use named constants
- Remove dead code and unused imports before committing

## Custom Coding Agent
This repository ships with a set of reusable Copilot prompt files (`.github/prompts/`) that form a **custom coding agent**:

| Prompt file | Purpose |
|---|---|
| `code-check.prompt.md` | Static analysis — find bugs, unused imports, style violations |
| `code-cleanup.prompt.md` | Automated cleanup — format, simplify, remove dead code |
| `code-review.prompt.md` | Thorough peer-style code review with actionable feedback |
| `full-code-audit.prompt.md` | All-in-one: check → clean → review in a single pass |

Run them from VS Code via **Chat → Attach prompt file**, or from the GitHub Copilot CLI:
```
gh copilot suggest -t shell "run the full-code-audit prompt on Task 1/data_cleaning_preprocessing.py"
```
