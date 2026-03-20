# Console

The `rd console` command opens an interactive terminal UI (TUI) for managing
demonstration metadata — reviewing recordings, marking them as success or
failure, and managing tasks and teachers.

```bash
rd console
```

## Workflow

During recording you mark each episode directly from the teaching hardware
(see [Marking demonstrations](recording.md#marking-demonstrations)).
The console is only needed when you want to correct a mistake — for example,
if you forgot to mark an episode or marked it incorrectly.

Only demonstrations marked as **success** are included when you run
`rd convert`. Pending and failure episodes are skipped automatically.

## Tabs

### Dashboard

Live counts across all demonstrations:

- **Total** — total number of recorded episodes
- **Pending** — not yet marked
- **Success** — marked as successful
- **Failure** — marked as failed
- **Converted** — already converted to dataset format

The lower half shows a per-task and per-teacher breakdown of the same counts.

### Demonstrations

A paginated table of all recorded episodes, sorted newest-first. Each row shows:

| Column | Description |
|---|---|
| ID | Database record ID |
| Task | Task name |
| Teacher | Teacher name |
| Status | `pending`, `success`, or `failure` |
| Converted | Whether the episode has been converted (`yes` / `no`) |
| Raw Path | Path to the raw recording directory |
| Created | Timestamp when the record was created |

**Actions** (navigate to a row with ↑ / ↓, then click the button):

| Button | Action |
|---|---|
| Mark Success | Set status to `success` |
| Mark Failure | Set status to `failure` |
| Delete | Permanently remove the DB entry (raw files are not deleted) |
| Update | Reassign the selected row to a different teacher or task |

### Teachers

Add, rename, or delete teacher records. Each teacher has a name and a count of
linked demonstrations.

### Tasks

A task defines what the robot should do during a demonstration. Each task has a
**name** (a short identifier) and a **language instruction** (a natural language
description shown to the operator during recording, e.g. `"Pick up the purrito"`).

Add, edit, or delete task definitions from this tab.

When you run `rd record`, you will be prompted to select a task. The selected
task name and instruction are embedded in every episode's lowdim data as
`language_task` and `language_prompt`.

## Key bindings

| Key | Action |
|---|---|
| `r` | Refresh all panes |
| `s` | Settings (DB path and record counts) |
| `?` | Help screen |
| `q` | Quit |
