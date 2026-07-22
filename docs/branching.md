# Branching and workflow

## Branches

| Branch | Role |
|---|---|
| `main` | Protected. **Never commit directly.** |
| `develop/v1` | Integration branch for the V1 milestone. Feature branches merge here. |
| `feature/*`, `test/*` | Short-lived, single-responsibility work branches. |

**`develop/v1` is a branch name that happens to contain a slash. It is not a namespace
and has no parent/child relationship to any other branch.** Every feature branch is
created from the latest `develop/v1`.

## Per-branch procedure

```bash
git checkout develop/v1
git pull                       # ensure it is current
git checkout -b feature/v1-<topic>
# implement only this branch's responsibility
pytest
git commit                     # logical, reviewable commits
```

After a branch is merged, the **next** feature branch is created from the newly updated
`develop/v1` — never from the previous feature branch.

A feature branch is merged into `develop/v1` only with **explicit authorisation from the
repository owner**. Do not self-merge.

## Reporting

On completing a branch, report:

- branch name
- files changed
- tests executed
- test result
- known limitations
- suggested PR title and description

## V1 branch sequence

| # | Branch | Responsibility |
|---|---|---|
| 1 | `feature/v1-spec-and-scaffold` | Repo inspection, docs, `CLAUDE.md`, package/test skeleton, packaging. No domain logic. |
| 2 | `feature/v1-schemas` | Typed schemas, JSON loading and validation, fixture files, schema tests. |
| 3 | `feature/v1-simulator-core` | Runtime state, network trace, battery model, termination, action validation. |
| 4 | `feature/v1-executors` | `BaseExecutor`, `ProfileExecutor`, `ReplayExecutor`, real executor stub. |
| 5 | `feature/v1-human-search-task` | Evidence tracker, task evaluator, synthetic prediction/ground-truth fixtures. |
| 6 | `feature/v1-policies` | Policy protocol, static baselines, rule-based baseline. |
| 7 | `feature/v1-metrics-and-cli` | Episode metrics, aggregate metrics, result serialization, CLI. |
| 8 | `test/v1-end-to-end` | Integration tests, deterministic fixture suite, docs cleanup, final V1 validation. |

Each branch implements **only** its own row. Work is not to be consolidated into one
large branch.

### Where `EpisodeRunner` lands

The loop itself is not in branch 3. It composes an `Executor` (branch 4), an
`EvidenceTracker` (branch 5), and a `Policy` (branch 6), so writing it earlier would mean
writing it against interfaces that do not exist yet and rewriting it three times.

Branch 3 delivers the components the loop drives — state transition, network, battery,
path, timing, termination, action validation — each independently tested. The runner that
sequences them belongs at the **start of branch 7**, where every dependency exists and the
composition root is being built anyway.

## Commit messages

Conventional commits where practical:

```
docs: define AeroIntentBench v1 architecture
feat: add typed benchmark schemas
feat: implement deterministic episode runner
feat: add profile and replay executors
feat: add human-search evidence evaluator
feat: add baseline configuration policies
feat: add metrics aggregation and CLI
test: add deterministic V1 integration suite
```
