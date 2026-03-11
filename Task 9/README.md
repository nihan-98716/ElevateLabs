# Task 9 — Custom Coding Agent for GitHub Copilot

This task explains how to build and use a **custom coding agent** that integrates with GitHub Copilot. The agent automates three quality-assurance steps — static analysis, code cleanup, and peer-style code review — through reusable prompt files stored in the repository.

---

## What the Agent Does

| Step | Prompt file | What it does |
|------|-------------|--------------|
| **Check** | `code-check.prompt.md` | Reads the code and reports bugs, style violations, anti-patterns, missing type hints, and missing docstrings. Makes **no changes**. |
| **Clean** | `code-cleanup.prompt.md` | Rewrites the file to fix every issue found in the check step while preserving behaviour exactly. |
| **Review** | `code-review.prompt.md` | Performs a formal peer-style review across correctness, robustness, design, performance, security, and documentation. |
| **All-in-one** | `full-code-audit.prompt.md` | Runs all three phases in a single pass: check → clean → review. |

---

## Step-by-Step Setup

### Prerequisites

| Tool | Minimum version | Install |
|------|-----------------|---------|
| Git | any | <https://git-scm.com> |
| VS Code | 1.90 | <https://code.visualstudio.com> |
| GitHub Copilot extension | latest | VS Code Marketplace |
| GitHub Copilot CLI extension | latest | `gh extension install github/gh-copilot` |
| GitHub CLI (`gh`) | 2.40 | <https://cli.github.com> |

---

### Step 1 — Clone the Repository

```bash
git clone https://github.com/nihan-98716/ElevateLabs.git
cd ElevateLabs
```

---

### Step 2 — Understand the File Structure

The agent lives entirely in `.github/`:

```
.github/
├── copilot-instructions.md        # Repository-level Copilot context (auto-loaded)
└── prompts/
    ├── code-check.prompt.md       # Phase 1 — static analysis
    ├── code-cleanup.prompt.md     # Phase 2 — automated cleanup
    ├── code-review.prompt.md      # Phase 3 — peer review
    └── full-code-audit.prompt.md  # All three phases in one
```

`copilot-instructions.md` is automatically read by GitHub Copilot for every chat interaction in this repository — no extra action needed.

The `.prompt.md` files are reusable agent definitions. Each file starts with a YAML front-matter block:

```yaml
---
mode: "agent"
description: "One-line summary shown in the VS Code prompt picker."
---
```

---

### Step 3 — Use the Agent in VS Code

1. Open any Python file in VS Code (e.g., `Task 1/data_cleaning_preprocessing.py`).
2. Open **GitHub Copilot Chat** (`Ctrl+Alt+I` / `Cmd+Alt+I`).
3. Click the **paperclip icon (Attach context)** → choose **Prompt file**.
4. Select one of the four prompt files from `.github/prompts/`.
5. Copilot will run the selected phase against your open file automatically.

Alternatively, type in the chat input:

```
@workspace /full-code-audit
```

to invoke the audit agent directly on the currently active file.

---

### Step 4 — Use the Agent from the GitHub Copilot CLI

Authenticate if you have not already:

```bash
gh auth login
gh extension install github/gh-copilot
```

Run a single phase from the terminal, replacing the file path as needed:

```bash
# Static analysis only
gh copilot suggest -t shell \
  "Using the instructions in .github/prompts/code-check.prompt.md, \
   analyze Task\ 1/data_cleaning_preprocessing.py and print the report."

# Cleanup only
gh copilot suggest -t shell \
  "Using the instructions in .github/prompts/code-cleanup.prompt.md, \
   clean Task\ 1/data_cleaning_preprocessing.py and print the result."

# Peer review only
gh copilot suggest -t shell \
  "Using the instructions in .github/prompts/code-review.prompt.md, \
   review Task\ 1/data_cleaning_preprocessing.py and print the report."

# Full audit (all three phases)
gh copilot suggest -t shell \
  "Using the instructions in .github/prompts/full-code-audit.prompt.md, \
   audit Task\ 1/data_cleaning_preprocessing.py and print all three phases."
```

---

### Step 5 — Add the Agent to a New Repository

To reuse this agent in any other Python project:

1. Copy the `.github/` folder into the root of the target repository.
2. Edit `.github/copilot-instructions.md` to reflect that project's conventions, libraries, and Python version.
3. Commit and push.

The prompt files themselves are generic and work without modification for any Python codebase.

---

### Step 6 — Customise the Prompts

Each prompt file is plain Markdown — edit it like any other file.

| Customisation | Where to change |
|---------------|-----------------|
| Add a new check rule | `code-check.prompt.md` → add a bullet under the relevant section |
| Change the output format | Any prompt → edit the `## Output Format` section |
| Target a different language | Replace Python-specific rules with language-appropriate ones |
| Adjust severity thresholds | `code-review.prompt.md` → edit the 🔴 / 🟡 / 🟢 criteria |
| Add a Phase 4 (e.g. test generation) | `full-code-audit.prompt.md` → append a new `## Phase 4` section |

---

## How It Works Internally

```
User opens file + attaches prompt
          │
          ▼
  Copilot reads .github/copilot-instructions.md   ← repo conventions & context
          │
          ▼
  Copilot reads selected .prompt.md file           ← agent instructions
          │
          ▼
  Copilot injects ${file} with the open file's content
          │
          ▼
  LLM generates output following the prompt's format rules
          │
          ▼
  Result shown in Copilot Chat panel / terminal
```

The `${file}` placeholder at the bottom of every prompt file is automatically replaced by VS Code with the full contents of the currently active editor file.

---

## Example Output

Running `full-code-audit.prompt.md` on `Task 1/data_cleaning_preprocessing.py` produces three sections:

**Phase 1** lists findings such as:
- 🟡 Line 54: `df['Age'].fillna(…, inplace=True)` — `inplace=True` is deprecated in pandas 2.x
- 🔵 `preprocess_titanic_data(url)` — missing return type hint `-> pd.DataFrame`

**Phase 2** returns the fully cleaned and annotated source file.

**Phase 3** gives a peer review table, e.g.:
- 🟡 Major: no input validation on `url` — function will raise an unhelpful `ParserError` on a bad URL
- 🟢 Minor: consider extracting the IQR outlier logic into a separate `remove_outliers(df, columns)` helper

---

## Files Created in This Task

```
.github/copilot-instructions.md
.github/prompts/code-check.prompt.md
.github/prompts/code-cleanup.prompt.md
.github/prompts/code-review.prompt.md
.github/prompts/full-code-audit.prompt.md
Task 9/README.md                          ← this file
```
