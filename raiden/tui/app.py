"""Textual TUI console for managing Raiden metadata."""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual import on
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Input,
    Label,
    Rule,
    Select,
    Static,
    TabbedContent,
    TabPane,
)

from raiden.db.database import get_db, reset_db

_PAGE_SIZE = 20

# ── CSS ───────────────────────────────────────────────────────────────────────

_CSS = """
Screen { background: #080d1a; }
Footer { background: #060b16; color: #7a9abf; }

/* ── Custom header ──────────────────────────────────────── */
_AppHeader {
    dock: top;
    background: #060b16;
    border-bottom: solid #00d4ff 40%;
    height: 3;
    padding: 0 2;
    align: left middle;
}
#header-title {
    width: 1fr;
    color: #00d4ff;
    text-style: bold;
    text-align: center;
    content-align: center middle;
}
/* ── Confirm modal ──────────────────────────────────────── */
_ConfirmScreen {
    align: center middle;
}
#confirm-dialog {
    background: #0d1526;
    border: solid #ff8c00;
    padding: 2 4;
    width: 50;
    height: auto;
}
#confirm-dialog Label {
    color: #ff8c00;
    text-style: bold;
    margin-bottom: 1;
}
#confirm-dialog .confirm-msg {
    color: #c8d8f0;
    margin-bottom: 2;
}
#confirm-buttons { align: center middle; height: 3; }
#confirm-yes { margin-right: 2; }

/* ── Selection bar ──────────────────────────────────────── */
.sel-bar { height: 3; align: left middle; margin-top: 0; }
.sel-bar Button { margin-right: 1; }
#demos-sel-label { color: #00d4ff; width: auto; margin-right: 2; text-style: bold; }

/* ── Settings modal ─────────────────────────────────────── */
_SettingsScreen {
    align: center middle;
}
#settings-dialog {
    background: #0d1526;
    border: solid #00d4ff;
    padding: 2 4;
    width: 60;
    height: auto;
}
#settings-dialog Label {
    color: #00d4ff;
    text-style: bold;
    margin-bottom: 1;
}
#settings-dialog .settings-row {
    height: 1;
    margin-bottom: 1;
}
#settings-dialog .settings-key { color: #7a9abf; width: 22; }
#settings-dialog .settings-val { color: #c8d8f0; width: 1fr; }
#settings-close {
    margin-top: 2;
    background: #0a2a4a;
    border: solid #00d4ff;
    color: #00d4ff;
}

/* ── Help modal ──────────────────────────────────────────── */
_HelpScreen {
    align: center middle;
}
#help-dialog {
    background: #0d1526;
    border: solid #00d4ff;
    padding: 2 4;
    width: 74;
    height: auto;
}
#help-dialog .help-heading {
    color: #00d4ff;
    text-style: bold;
    margin-top: 1;
    margin-bottom: 0;
}
#help-dialog .help-body {
    color: #c8d8f0;
    margin-bottom: 1;
}
#help-dialog .help-key { color: #7a9abf; width: 26; }
#help-dialog .help-desc { color: #c8d8f0; width: 1fr; }
#help-dialog .help-row { height: 1; margin-bottom: 0; }
#help-close {
    margin-top: 2;
    background: #0a2a4a;
    border: solid #00d4ff;
    color: #00d4ff;
}

TabbedContent { background: #080d1a; height: 1fr; }
TabbedContent ContentSwitcher { height: 1fr; }
TabPane { background: #080d1a; padding: 1 2; height: 1fr; }

/* ── Dashboard ──────────────────────────────────────────── */
.stat-row { height: 5; margin-bottom: 1; }
.stat-box {
    width: 1fr;
    height: 5;
    background: #111c35;
    border: solid #00d4ff;
    text-align: center;
    content-align: center middle;
    color: #00d4ff;
    text-style: bold;
    margin: 0 1;
}
.breakdown-row { height: 1fr; }
.breakdown-pane { width: 1fr; margin: 0 1; }
.section-title { color: #00d4ff; text-style: bold; height: 1; }

/* ── Tables ─────────────────────────────────────────────── */
DataTable {
    background: #0d1526;
    border: solid #00d4ff 20%;
    height: 1fr;
}
DataTable > .datatable--header {
    background: #0d1e3a;
    color: #00d4ff;
    text-style: bold;
}
DataTable > .datatable--cursor {
    background: #1a3a6e;
    color: #c8d8f0;
}

/* ── Forms ──────────────────────────────────────────────── */
.action-bar { height: 3; margin-top: 1; align: left middle; }
.action-bar Button { margin-right: 1; }

.form-row { height: 5; margin-top: 1; align: left middle; }
.form-row Input {
    width: 28;
    margin-right: 1;
    background: #0d1526;
    border: solid #00d4ff 30%;
    color: #c8d8f0;
}
.form-row Input.-focus { border: solid #00d4ff; }
.form-row Select { width: 32; margin-right: 1; }
.form-row Button { margin-right: 1; }

/* ── Pagination ─────────────────────────────────────────── */
.pagination-bar { height: 3; align: left middle; margin-top: 0; }
.page-label { color: #7a9abf; margin: 0 2; width: auto; }

/* ── Button variants ────────────────────────────────────── */
Button.success   { background: #1a4a1a; border: solid #00ff44; color: #00ff44; }
Button.failure   { background: #4a1a1a; border: solid #ff4444; color: #ff4444; }
Button.danger    { background: #3a1a0a; border: solid #ff8c00; color: #ff8c00; }
Button.primary   { background: #0a2a4a; border: solid #00d4ff; color: #00d4ff; }
Button.secondary { background: #1a1a3a; border: solid #7a9abf; color: #7a9abf; }
"""


# ── Helpers ───────────────────────────────────────────────────────────────────


def _fmt_dt(iso: str) -> str:
    return iso[:16].replace("T", " ") if iso else ""


def _select_val(sel: Select):
    """Return the selected value, or None if blank.

    Uses both identity and type-name checks so it works regardless of whether
    the installed Textual version uses a singleton or not for NoSelection.
    """
    v = sel.value
    if v is Select.BLANK:
        return None
    if type(v).__name__ == "NoSelection":
        return None
    return v


# ── Custom header ─────────────────────────────────────────────────────────────


class _AppHeader(Horizontal):
    def compose(self) -> ComposeResult:
        yield Static("Raiden Console", id="header-title")


# ── Settings modal ────────────────────────────────────────────────────────────


class _ConfirmScreen(ModalScreen[bool]):
    """Generic yes/no confirmation modal."""

    def __init__(self, message: str, verb: str = "Confirm") -> None:
        super().__init__()
        self._message = message
        self._verb = verb

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-dialog"):
            yield Label("Warning")
            yield Static(self._message, classes="confirm-msg")
            with Horizontal(id="confirm-buttons"):
                yield Button(self._verb, id="confirm-yes", classes="danger")
                yield Button("Cancel", id="confirm-no", classes="secondary")

    @on(Button.Pressed, "#confirm-yes")
    def _confirm(self, event: Button.Pressed) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#confirm-no")
    def _cancel(self, event: Button.Pressed) -> None:
        self.dismiss(False)


class _SettingsScreen(ModalScreen):
    def compose(self) -> ComposeResult:
        from raiden._config import DB_DIR

        db = get_db()
        db_dir = DB_DIR.resolve()

        n_demos = len(db.get_demonstrations())
        n_teachers = len(db.get_teachers())
        n_tasks = len(db.get_tasks())
        n_calib = len(db.get_calibration_results())
        n_cfg = len(db.get_camera_configs())

        with Vertical(id="settings-dialog"):
            yield Label("Settings")
            with Horizontal(classes="settings-row"):
                yield Static("DB directory", classes="settings-key")
                yield Static(str(db_dir), classes="settings-val")
            with Horizontal(classes="settings-row"):
                yield Static("Demonstrations", classes="settings-key")
                yield Static(str(n_demos), classes="settings-val")
            with Horizontal(classes="settings-row"):
                yield Static("Teachers", classes="settings-key")
                yield Static(str(n_teachers), classes="settings-val")
            with Horizontal(classes="settings-row"):
                yield Static("Tasks", classes="settings-key")
                yield Static(str(n_tasks), classes="settings-val")
            with Horizontal(classes="settings-row"):
                yield Static("Calibration results", classes="settings-key")
                yield Static(str(n_calib), classes="settings-val")
            with Horizontal(classes="settings-row"):
                yield Static("Camera configs", classes="settings-key")
                yield Static(str(n_cfg), classes="settings-val")
            with Horizontal(classes="settings-row"):
                yield Static("Keybindings", classes="settings-key")
                yield Static("r  Refresh    q  Quit", classes="settings-val")
            yield Button("Close", id="settings-close")

    @on(Button.Pressed, "#settings-close")
    def _close(self, event: Button.Pressed) -> None:
        self.dismiss()


class _HelpScreen(ModalScreen):
    def compose(self) -> ComposeResult:
        with Vertical(id="help-dialog"):
            yield Label("Raiden Console — Help")

            yield Static("Workflow", classes="help-heading")
            yield Static(
                "Mark each episode as Success or Failure directly from the teaching\n"
                "hardware during recording. Open the console only to correct mistakes.\n\n"
                "Only successful demonstrations are converted when you run  rd convert.",
                classes="help-body",
            )

            yield Static("Marking demonstrations", classes="help-heading")
            yield Static(
                "During recording:\n"
                "  Left pedal / leader button          Start or stop recording\n"
                "  Middle pedal / top leader button    Mark as Success\n"
                "  Right pedal / bottom leader button  Mark as Failure\n\n"
                "In the Demonstrations tab:\n"
                "  ↑ / ↓                Navigate rows\n"
                "  Space                Toggle selection on the current row\n"
                "  Select All           Select all rows on the current page\n"
                "  Clear                Deselect all rows\n"
                "  Mark Success/Failure Apply to all selected rows (or cursor row)\n"
                "  Update               Reassign teacher/task for all selected rows\n"
                "  Delete               Delete all selected rows (or cursor row)",
                classes="help-body",
            )

            yield Static("Converting", classes="help-heading")
            yield Static(
                "Run  rd convert  after recording to extract PNG frames and depth maps.\n"
                "Only demonstrations with status = success are converted.\n"
                "Pending and failure demonstrations are skipped.",
                classes="help-body",
            )

            yield Static("Tabs", classes="help-heading")
            yield Static(
                "Dashboard       Live counts and per-task / per-teacher breakdown\n"
                "Demonstrations  Full list of recorded episodes with status and paths\n"
                "Teachers        Manage teacher names linked to demonstrations\n"
                "Tasks           Manage task names and language instructions",
                classes="help-body",
            )

            yield Static("Key bindings", classes="help-heading")
            yield Static(
                "r   Refresh all panes\n"
                "s   Settings (DB path and record counts)\n"
                "?   This help screen\n"
                "q   Quit",
                classes="help-body",
            )

            yield Button("Close", id="help-close")

    @on(Button.Pressed, "#help-close")
    def _close(self, event: Button.Pressed) -> None:
        self.dismiss()


def _selected_id(table: DataTable) -> int | None:
    if table.row_count == 0:
        return None
    try:
        row = table.get_row_at(table.cursor_row)
        return int(str(row[0]))
    except Exception:
        return None


# ── App ───────────────────────────────────────────────────────────────────────


class RaidenConsole(App):
    """Interactive console for managing Raiden demonstrations and metadata."""

    CSS = _CSS
    TITLE = "Raiden Console"
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh", "Refresh"),
        ("s", "open_settings", "Settings"),
        ("question_mark", "help", "Help"),
    ]

    def compose(self) -> ComposeResult:
        yield _AppHeader()
        with TabbedContent():
            with TabPane("Dashboard", id="tab-dashboard"):
                yield _DashboardPane()
            with TabPane("Demonstrations", id="tab-demonstrations"):
                yield _DemonstrationsPane()
            with TabPane("Teachers", id="tab-teachers"):
                yield _TeachersPane()
            with TabPane("Tasks", id="tab-tasks"):
                yield _TasksPane()
        yield Footer()

    def on_mount(self) -> None:
        self.set_interval(2, self.action_refresh)

    def action_open_settings(self) -> None:
        self.push_screen(_SettingsScreen())

    def action_help(self) -> None:
        self.push_screen(_HelpScreen())

    def action_refresh(self) -> None:
        import json as _json  # noqa: PLC0415

        for cls in (_DashboardPane, _DemonstrationsPane, _TeachersPane, _TasksPane):
            try:
                self.query_one(cls).safe_refresh()  # type: ignore[attr-defined]
            except _json.JSONDecodeError:
                # DB file corrupt — repair and reload the singleton, then retry
                try:
                    get_db().repair()
                    reset_db()
                    self.query_one(cls).safe_refresh()  # type: ignore[attr-defined]
                except Exception:
                    pass
            except Exception:
                pass

    @on(TabbedContent.TabActivated)
    def _on_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        self.action_refresh()


# ── Dashboard pane ─────────────────────────────────────────────────────────────


class _DashboardPane(Vertical):
    def compose(self) -> ComposeResult:
        with Horizontal(classes="stat-row"):
            yield Static("Total\n\n0", id="stat-total", classes="stat-box")
            yield Static("Pending\n\n0", id="stat-pending", classes="stat-box")
            yield Static("Success\n\n0", id="stat-success", classes="stat-box")
            yield Static("Failure\n\n0", id="stat-failure", classes="stat-box")
            yield Static("Converted\n\n0", id="stat-converted", classes="stat-box")
        with Horizontal(classes="breakdown-row"):
            with Vertical(classes="breakdown-pane"):
                yield Static("By Task", classes="section-title")
                yield DataTable(id="by-task-table")
            with Vertical(classes="breakdown-pane"):
                yield Static("By Teacher", classes="section-title")
                yield DataTable(id="by-teacher-table")

    def on_mount(self) -> None:
        task_tbl = self.query_one("#by-task-table", DataTable)
        task_tbl.add_columns(
            "Task", "Total", "Success", "Failure", "Pending", "Converted"
        )
        teacher_tbl = self.query_one("#by-teacher-table", DataTable)
        teacher_tbl.add_columns(
            "Teacher", "Total", "Success", "Failure", "Pending", "Converted"
        )
        self.safe_refresh()

    def safe_refresh(self) -> None:
        try:
            self.refresh_data()
        except Exception:
            pass

    def refresh_data(self) -> None:
        db = get_db()
        demos = db.get_demonstrations()
        tasks = db.get_tasks()
        teachers = db.get_teachers()

        n_total = len(demos)
        by_status: dict[str, int] = {}
        n_converted = 0
        for d in demos:
            s = d.get("status", "pending")
            by_status[s] = by_status.get(s, 0) + 1
            if d.get("converted", False):
                n_converted += 1

        self.query_one("#stat-total", Static).update(f"Total\n\n{n_total}")
        self.query_one("#stat-pending", Static).update(
            f"Pending\n\n{by_status.get('pending', 0)}"
        )
        self.query_one("#stat-success", Static).update(
            f"Success\n\n{by_status.get('success', 0)}"
        )
        self.query_one("#stat-failure", Static).update(
            f"Failure\n\n{by_status.get('failure', 0)}"
        )
        self.query_one("#stat-converted", Static).update(f"Converted\n\n{n_converted}")

        # Per-task breakdown (status counts + converted count)
        per_task_status: dict[int, dict[str, int]] = {}
        per_task_conv: dict[int, int] = {}
        for d in demos:
            tid = d.get("task_id")
            if tid is not None:
                s = d.get("status", "pending")
                per_task_status.setdefault(tid, {})
                per_task_status[tid][s] = per_task_status[tid].get(s, 0) + 1
                if d.get("converted", False):
                    per_task_conv[tid] = per_task_conv.get(tid, 0) + 1

        task_tbl = self.query_one("#by-task-table", DataTable)
        task_tbl.clear()
        for t in tasks:
            c = per_task_status.get(t["id"], {})
            task_tbl.add_row(
                t["name"],
                str(sum(c.values())),
                str(c.get("success", 0)),
                str(c.get("failure", 0)),
                str(c.get("pending", 0)),
                str(per_task_conv.get(t["id"], 0)),
                key=str(t["id"]),
            )

        # Per-teacher breakdown
        per_teacher_status: dict[int, dict[str, int]] = {}
        per_teacher_conv: dict[int, int] = {}
        for d in demos:
            tid = d.get("teacher_id")
            if tid is not None:
                s = d.get("status", "pending")
                per_teacher_status.setdefault(tid, {})
                per_teacher_status[tid][s] = per_teacher_status[tid].get(s, 0) + 1
                if d.get("converted", False):
                    per_teacher_conv[tid] = per_teacher_conv.get(tid, 0) + 1

        teacher_tbl = self.query_one("#by-teacher-table", DataTable)
        teacher_tbl.clear()
        for t in teachers:
            c = per_teacher_status.get(t["id"], {})
            teacher_tbl.add_row(
                t["name"],
                str(sum(c.values())),
                str(c.get("success", 0)),
                str(c.get("failure", 0)),
                str(c.get("pending", 0)),
                str(per_teacher_conv.get(t["id"], 0)),
                key=str(t["id"]),
            )


# ── Demonstrations table with Space-to-toggle ─────────────────────────────────


class _DemoTable(DataTable):
    """DataTable subclass that emits RowToggled when Space is pressed."""

    class RowToggled(Message):
        def __init__(self, row_index: int) -> None:
            super().__init__()
            self.row_index = row_index

    def on_key(self, event) -> None:
        if event.key == "space" and self.row_count > 0:
            self.post_message(self.RowToggled(self.cursor_row))
            event.stop()


# ── Demonstrations pane ───────────────────────────────────────────────────────


class _DemonstrationsPane(Vertical):
    def __init__(self) -> None:
        super().__init__()
        self._page = 0
        self._selected_keys: set[str] = set()

    def compose(self) -> ComposeResult:
        yield _DemoTable(id="demos-table", cursor_type="row")
        with Horizontal(classes="sel-bar"):
            yield Static("", id="demos-sel-label")
            yield Button("Select All", id="demos-sel-all", classes="secondary")
            yield Button("Clear", id="demos-sel-clear", classes="secondary")
        yield Rule(id="demos-divider")
        with Horizontal(classes="pagination-bar", id="demos-pagination"):
            yield Button("< Prev", id="demos-prev", classes="secondary")
            yield Static("Page 1 / 1", id="demos-page-label", classes="page-label")
            yield Button("Next >", id="demos-next", classes="secondary")
        with Horizontal(classes="action-bar"):
            yield Button("Mark Success", id="demo-success", classes="success")
            yield Button("Mark Failure", id="demo-failure", classes="failure")
            yield Button("Delete", id="demo-delete", classes="danger")
        # Reassign teacher / task for all selected demos
        with Horizontal(classes="form-row"):
            yield Select([], id="demo-teacher-select", prompt="Reassign teacher...")
            yield Select([], id="demo-task-select", prompt="Reassign task...")
            yield Button("Update", id="demo-update", classes="primary")

    def on_mount(self) -> None:
        table = self.query_one("#demos-table", _DemoTable)
        table.add_column("", key="sel", width=2)
        table.add_columns(
            "ID", "Task", "Teacher", "Status", "Converted", "Raw Path", "Created"
        )
        self.safe_refresh()

    def safe_refresh(self) -> None:
        try:
            self.refresh_data()
        except Exception:
            pass

    def refresh_data(self) -> None:
        db = get_db()
        table = self.query_one("#demos-table", _DemoTable)

        old_key: str | None = None
        if table.row_count > 0:
            try:
                old_key = str(table.get_row_at(table.cursor_row)[1])
            except Exception:
                pass

        table.clear()

        task_map = {t["id"]: t["name"] for t in db.get_tasks()}
        teacher_map = {t["id"]: t["name"] for t in db.get_teachers()}

        all_demos = sorted(
            db.get_demonstrations(),
            key=lambda d: d.get("created_at", ""),
            reverse=True,
        )
        total = len(all_demos)
        total_pages = max(1, (total + _PAGE_SIZE - 1) // _PAGE_SIZE)
        self._page = min(self._page, total_pages - 1)

        page_demos = all_demos[self._page * _PAGE_SIZE : (self._page + 1) * _PAGE_SIZE]
        for d in page_demos:
            row_key = str(d["id"])
            table.add_row(
                "●" if row_key in self._selected_keys else " ",
                row_key,
                task_map.get(d.get("task_id"), "?"),
                teacher_map.get(d.get("teacher_id"), "?"),
                d.get("status", "pending"),
                "yes" if d.get("converted", False) else "no",
                d.get("raw_data_path", ""),
                _fmt_dt(d.get("created_at", "")),
                key=row_key,
            )

        show_pagination = total_pages > 1
        self.query_one("#demos-pagination").display = show_pagination
        self.query_one("#demos-divider").display = not show_pagination
        if show_pagination:
            self.query_one("#demos-page-label", Static).update(
                f"Page {self._page + 1} / {total_pages}  ({total} total)"
            )

        teachers = [(t["name"], t["id"]) for t in db.get_teachers()]
        tasks = [(t["name"], t["id"]) for t in db.get_tasks()]
        teacher_sel = self.query_one("#demo-teacher-select", Select)
        task_sel = self.query_one("#demo-task-select", Select)
        prev_teacher = _select_val(teacher_sel)
        prev_task = _select_val(task_sel)
        teacher_sel.set_options(teachers)
        task_sel.set_options(tasks)
        if prev_teacher is not None:
            teacher_sel.value = prev_teacher
        if prev_task is not None:
            task_sel.value = prev_task

        if old_key is not None:
            try:
                table.move_cursor(row=table.get_row_index(old_key))
            except Exception:
                pass

        self._update_sel_label()

    # ── selection helpers ──────────────────────────────────────────────────

    def _cursor_id(self) -> int | None:
        table = self.query_one("#demos-table", _DemoTable)
        if table.row_count == 0:
            return None
        try:
            return int(str(table.get_row_at(table.cursor_row)[1]))
        except Exception:
            return None

    def _selected_ids(self) -> list[int]:
        """Return selected demo IDs, falling back to the cursor row if none selected."""
        if self._selected_keys:
            return [int(k) for k in self._selected_keys]
        cid = self._cursor_id()
        return [cid] if cid is not None else []

    def _update_sel_label(self) -> None:
        n = len(self._selected_keys)
        text = f"{n} selected  (Space to toggle)" if n else "Space to select rows"
        self.query_one("#demos-sel-label", Static).update(text)

    def _advance_cursor(self) -> None:
        """Move the cursor to the next row (used after single-row actions)."""
        table = self.query_one("#demos-table", _DemoTable)
        if table.row_count > 0:
            table.move_cursor(row=min(table.cursor_row + 1, table.row_count - 1))

    @on(_DemoTable.RowToggled)
    def _on_row_toggled(self, event: _DemoTable.RowToggled) -> None:
        table = self.query_one("#demos-table", _DemoTable)
        try:
            row_key = str(table.get_row_at(event.row_index)[1])
            if row_key in self._selected_keys:
                self._selected_keys.discard(row_key)
                table.update_cell(row_key, "sel", " ")
            else:
                self._selected_keys.add(row_key)
                table.update_cell(row_key, "sel", "●")
            self._update_sel_label()
            self._advance_cursor()
        except Exception:
            pass

    @on(Button.Pressed, "#demos-sel-all")
    def _select_all(self, event: Button.Pressed) -> None:
        table = self.query_one("#demos-table", _DemoTable)
        for i in range(table.row_count):
            try:
                row_key = str(table.get_row_at(i)[1])
                self._selected_keys.add(row_key)
                table.update_cell(row_key, "sel", "●")
            except Exception:
                pass
        self._update_sel_label()

    @on(Button.Pressed, "#demos-sel-clear")
    def _clear_selection(self, event: Button.Pressed) -> None:
        table = self.query_one("#demos-table", _DemoTable)
        for i in range(table.row_count):
            try:
                row_key = str(table.get_row_at(i)[1])
                table.update_cell(row_key, "sel", " ")
            except Exception:
                pass
        self._selected_keys.clear()
        self._update_sel_label()

    # ── pagination ─────────────────────────────────────────────────────────

    @on(Button.Pressed, "#demos-prev")
    def _prev_page(self, event: Button.Pressed) -> None:
        if self._page > 0:
            self._page -= 1
            self.refresh_data()

    @on(Button.Pressed, "#demos-next")
    def _next_page(self, event: Button.Pressed) -> None:
        db = get_db()
        total = len(db.get_demonstrations())
        total_pages = max(1, (total + _PAGE_SIZE - 1) // _PAGE_SIZE)
        if self._page < total_pages - 1:
            self._page += 1
            self.refresh_data()

    # ── actions ────────────────────────────────────────────────────────────

    @on(Button.Pressed, "#demo-success")
    def _mark_success(self, event: Button.Pressed) -> None:
        ids = self._selected_ids()
        if not ids:
            return
        single = not self._selected_keys
        db = get_db()
        for demo_id in ids:
            db.update_demonstration(demo_id, status="success")
        self.refresh_data()
        if single:
            self._advance_cursor()

    @on(Button.Pressed, "#demo-failure")
    def _mark_failure(self, event: Button.Pressed) -> None:
        ids = self._selected_ids()
        if not ids:
            return
        single = not self._selected_keys
        db = get_db()
        for demo_id in ids:
            db.update_demonstration(demo_id, status="failure")
        self.refresh_data()
        if single:
            self._advance_cursor()

    @on(Button.Pressed, "#demo-update")
    def _update_demo(self, event: Button.Pressed) -> None:
        ids = self._selected_ids()
        if not ids:
            return
        single = not self._selected_keys
        kwargs: dict = {}
        teacher_val = _select_val(self.query_one("#demo-teacher-select", Select))
        task_val = _select_val(self.query_one("#demo-task-select", Select))
        if teacher_val is not None:
            kwargs["teacher_id"] = teacher_val
        if task_val is not None:
            kwargs["task_id"] = task_val
        if kwargs:
            db = get_db()
            for demo_id in ids:
                db.update_demonstration(demo_id, **kwargs)
            self.refresh_data()
            if single:
                self._advance_cursor()

    @on(Button.Pressed, "#demo-delete")
    def _delete_demo(self, event: Button.Pressed) -> None:
        ids = self._selected_ids()
        if not ids:
            return
        single = not self._selected_keys
        n = len(ids)
        msg = (
            f"Delete {n} demonstrations?"
            if n > 1
            else f"Delete demonstration #{ids[0]}?"
        )

        def _on_confirm(confirmed: bool) -> None:
            if confirmed:
                db = get_db()
                for demo_id in ids:
                    db.delete_demonstration(demo_id)
                self._selected_keys -= {str(i) for i in ids}
                self.refresh_data()
                if single:
                    self._advance_cursor()

        self.app.push_screen(_ConfirmScreen(msg, verb="Delete"), _on_confirm)


# ── Teachers pane ─────────────────────────────────────────────────────────────


class _TeachersPane(Vertical):
    def compose(self) -> ComposeResult:
        yield DataTable(id="teachers-table", cursor_type="row")
        with Horizontal(classes="form-row"):
            yield Input(placeholder="Name", id="teacher-name-input")
            yield Button("Add", id="teacher-add", classes="primary")
            yield Button("Update", id="teacher-update", classes="secondary")
            yield Button("Delete", id="teacher-delete", classes="danger")

    def on_mount(self) -> None:
        table = self.query_one("#teachers-table", DataTable)
        table.add_columns("ID", "Name", "Demos", "Created")
        self.safe_refresh()

    def safe_refresh(self) -> None:
        try:
            self.refresh_data()
        except Exception:
            pass

    def refresh_data(self) -> None:
        db = get_db()
        table = self.query_one("#teachers-table", DataTable)

        old_key: str | None = None
        if table.row_count > 0:
            try:
                old_key = str(table.get_row_at(table.cursor_row)[0])
            except Exception:
                pass

        table.clear()

        demos = db.get_demonstrations()
        demo_counts: dict[int, int] = {}
        for d in demos:
            tid = d.get("teacher_id")
            if tid is not None:
                demo_counts[tid] = demo_counts.get(tid, 0) + 1

        for t in db.get_teachers():
            table.add_row(
                str(t["id"]),
                t["name"],
                str(demo_counts.get(t["id"], 0)),
                _fmt_dt(t.get("created_at", "")),
                key=str(t["id"]),
            )

        if old_key is not None:
            try:
                table.move_cursor(row=table.get_row_index(old_key))
            except Exception:
                pass

    @on(Button.Pressed, "#teacher-add")
    def _add_teacher(self, event: Button.Pressed) -> None:
        name = self.query_one("#teacher-name-input", Input).value.strip()
        if not name:
            return
        db = get_db()
        if db.get_teacher_by_name(name) is None:
            db.add_teacher(name)
        self.query_one("#teacher-name-input", Input).value = ""
        self.refresh_data()

    @on(Button.Pressed, "#teacher-update")
    def _update_teacher(self, event: Button.Pressed) -> None:
        teacher_id = _selected_id(self.query_one("#teachers-table", DataTable))
        name = self.query_one("#teacher-name-input", Input).value.strip()
        if teacher_id is not None and name:
            get_db().update_teacher(teacher_id, name)
            self.query_one("#teacher-name-input", Input).value = ""
            self.refresh_data()

    @on(Button.Pressed, "#teacher-delete")
    def _delete_teacher(self, event: Button.Pressed) -> None:
        table = self.query_one("#teachers-table", DataTable)
        teacher_id = _selected_id(table)
        if teacher_id is None:
            return
        try:
            name = str(table.get_row_at(table.cursor_row)[1])
        except Exception:
            name = f"#{teacher_id}"

        def _on_confirm(confirmed: bool) -> None:
            if confirmed:
                get_db().delete_teacher(teacher_id)
                self.query_one("#teacher-name-input", Input).value = ""
                self.refresh_data()

        self.app.push_screen(
            _ConfirmScreen(
                f'Delete teacher "{name}"?\nAll linked demonstrations will lose their teacher reference.'
            ),
            _on_confirm,
        )


# ── Tasks pane ────────────────────────────────────────────────────────────────


class _TasksPane(Vertical):
    def __init__(self) -> None:
        super().__init__()
        self._page = 0

    def compose(self) -> ComposeResult:
        yield DataTable(id="tasks-table", cursor_type="row")
        yield Rule(id="tasks-divider")
        with Horizontal(classes="pagination-bar", id="tasks-pagination"):
            yield Button("< Prev", id="tasks-prev", classes="secondary")
            yield Static("Page 1 / 1", id="tasks-page-label", classes="page-label")
            yield Button("Next >", id="tasks-next", classes="secondary")
        with Horizontal(classes="form-row"):
            yield Input(placeholder="Name", id="task-name-input")
            yield Input(placeholder="Instruction", id="task-instruction-input")
            yield Button("Add", id="task-add", classes="primary")
            yield Button("Update", id="task-update", classes="secondary")
            yield Button("Delete", id="task-delete", classes="danger")

    def on_mount(self) -> None:
        table = self.query_one("#tasks-table", DataTable)
        table.add_columns("ID", "Name", "Instruction", "Demos", "Created")
        self.safe_refresh()

    def safe_refresh(self) -> None:
        try:
            self.refresh_data()
        except Exception:
            pass

    def refresh_data(self) -> None:
        db = get_db()
        table = self.query_one("#tasks-table", DataTable)

        old_key: str | None = None
        if table.row_count > 0:
            try:
                old_key = str(table.get_row_at(table.cursor_row)[0])
            except Exception:
                pass

        table.clear()

        demos = db.get_demonstrations()
        demo_counts: dict[int, int] = {}
        for d in demos:
            tid = d.get("task_id")
            if tid is not None:
                demo_counts[tid] = demo_counts.get(tid, 0) + 1

        all_tasks = sorted(
            db.get_tasks(),
            key=lambda t: t.get("created_at", ""),
            reverse=True,
        )
        total = len(all_tasks)
        total_pages = max(1, (total + _PAGE_SIZE - 1) // _PAGE_SIZE)
        self._page = min(self._page, total_pages - 1)

        page_tasks = all_tasks[self._page * _PAGE_SIZE : (self._page + 1) * _PAGE_SIZE]
        for t in page_tasks:
            table.add_row(
                str(t["id"]),
                t["name"],
                t.get("instruction", ""),
                str(demo_counts.get(t["id"], 0)),
                _fmt_dt(t.get("created_at", "")),
                key=str(t["id"]),
            )

        show_pagination = total_pages > 1
        self.query_one("#tasks-pagination").display = show_pagination
        self.query_one("#tasks-divider").display = not show_pagination
        if show_pagination:
            self.query_one("#tasks-page-label", Static).update(
                f"Page {self._page + 1} / {total_pages}  ({total} total)"
            )

        if old_key is not None:
            try:
                table.move_cursor(row=table.get_row_index(old_key))
            except Exception:
                pass

    @on(Button.Pressed, "#tasks-prev")
    def _prev_page(self, event: Button.Pressed) -> None:
        if self._page > 0:
            self._page -= 1
            self.refresh_data()

    @on(Button.Pressed, "#tasks-next")
    def _next_page(self, event: Button.Pressed) -> None:
        db = get_db()
        total = len(db.get_tasks())
        total_pages = max(1, (total + _PAGE_SIZE - 1) // _PAGE_SIZE)
        if self._page < total_pages - 1:
            self._page += 1
            self.refresh_data()

    @on(Button.Pressed, "#task-add")
    def _add_task(self, event: Button.Pressed) -> None:
        name = self.query_one("#task-name-input", Input).value.strip()
        instruction = self.query_one("#task-instruction-input", Input).value.strip()
        if not name or not instruction:
            return
        db = get_db()
        if db.get_task_by_name(name) is None:
            db.add_task(name, instruction)
        self.query_one("#task-name-input", Input).value = ""
        self.query_one("#task-instruction-input", Input).value = ""
        self.refresh_data()

    @on(Button.Pressed, "#task-update")
    def _update_task(self, event: Button.Pressed) -> None:
        task_id = _selected_id(self.query_one("#tasks-table", DataTable))
        name = self.query_one("#task-name-input", Input).value.strip()
        instruction = self.query_one("#task-instruction-input", Input).value.strip()
        if task_id is not None and name and instruction:
            get_db().update_task(task_id, name, instruction)
            self.query_one("#task-name-input", Input).value = ""
            self.query_one("#task-instruction-input", Input).value = ""
            self.refresh_data()

    @on(Button.Pressed, "#task-delete")
    def _delete_task(self, event: Button.Pressed) -> None:
        table = self.query_one("#tasks-table", DataTable)
        task_id = _selected_id(table)
        if task_id is None:
            return
        try:
            name = str(table.get_row_at(table.cursor_row)[1])
        except Exception:
            name = f"#{task_id}"

        def _on_confirm(confirmed: bool) -> None:
            if confirmed:
                get_db().delete_task(task_id)
                self.query_one("#task-name-input", Input).value = ""
                self.query_one("#task-instruction-input", Input).value = ""
                self.refresh_data()

        self.app.push_screen(
            _ConfirmScreen(
                f'Delete task "{name}"?\nAll linked demonstrations will lose their task reference.'
            ),
            _on_confirm,
        )
