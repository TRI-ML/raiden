# Tasks

A task defines what the robot should do during a demonstration. Each task has a
**name** (a short identifier) and a **language instruction** (a natural language
description shown to the operator during recording).

## Managing tasks

Tasks are created and edited in the [console](console.md) under the **Tasks**
tab. Open it with:

```bash
rd console
```

From the Tasks tab you can add, rename, or delete tasks.

## Using a task during recording

When you run `rd record`, you will be prompted to select a task from the list.
The selected task name and instruction are embedded in every episode's lowdim
data as `language_task` and `language_prompt`.
