# Engram revert plan

Memory MCP promises a hot cache injected into Claude's context at 0ms with no tool call. As of 2026-09-11 no code path delivers it. This program makes the promise true with a hook, makes promotion depend on real use, and deletes the mining machinery that was built because the promise was never wired. Five PRs, in this order. PR-1 and PR-2 are independent and start together. PR-3 stacks on PR-2. PR-4 stacks on PR-3. PR-5 stacks on PR-4. The operator merges every PR. Nobody else merges.

Rule the program enforces. Every memory that reaches Claude arrives through a hook printing to stdout, and every promotion or demotion is driven by whether an injected memory was used.

## How to read this

One box is one unit of work. Every box names the evidence that checks it. Check a box only when its evidence exists, which means a file, a log line, a test run, or a SHA. Do not check a box because you believe the work is done. The body is a how-to. The appendices explain.

The program runs `skills/poteto-mode/playbooks/autopilot-stack.md` from the installed pstack plugin. The operator lands the stack. PR-1 targets `main`. PR-2 targets `main`. PR-3 targets the PR-2 branch. PR-4 targets the PR-3 branch. PR-5 targets the PR-4 branch.

Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

## Program checklist

Read this whole section before you touch a file. Follow it literally. Every box names the evidence that checks it.

### Arm the program

- [ ] State the protocol and this plan to the operator, then stop. Start execution only on the operator's explicit go. The go for this program was given on 2026-09-11 in the same message that asked for the plan.
- [ ] On the go, write this exact text into the standing orders and restate it in your todolist. "Plan at `.planning/superpowers/plans/2026-09-11-engram-revert-plan.md` on branch `engram-revert-plan`. PR-1 and PR-2 first from `main`, then PR-3 on PR-2, PR-4 on PR-3, PR-5 on PR-4. Tests alone are not sufficient verification. The operator merges. Done when every PR has a clean verdict at its head SHA and the stack is linear."
- [ ] Read these from the installed plugin at program start. Re-read them at every tick.
  - [ ] `skills/poteto-mode/playbooks/autopilot-stack.md`
  - [ ] `skills/swarm/SKILL.md`
  - [ ] `skills/poteto-mode/playbooks/opening-a-pr.md`
  - [ ] `skills/show-me-your-work/SKILL.md`
- [ ] Arm the 30-minute audit tick as a real `/loop` in dynamic mode, which schedules its own wake-up rather than blocking on a sleep. Never leave the cadence to memory.
- [ ] Use this tick prompt, verbatim. "Re-read the execution playbook from the installed plugin and the standing objective. Audit the operation against both and fix drift in this tick. Probe every active lane and judge progress by side effects only. Stand down a stuck lane and dispatch its replacement now. Then send the operator a status message, whether or not anything changed, with the queue table of PR, owner, state, and head SHA, the verdicts since the last tick, what merged, open operator gates, and blockers."
- [ ] On the operator's hold or stand-down, send every owner a zero-writes order at once.

### Spawn owners

- [ ] Spawn one owner per PR with `subagent_type: "pstack:poteto-agent"` and the opus model, in its own worktree, with the plan path and its PR id. The owner reads only its own section plus the Program checklist.
- [ ] Follow this dependency graph.
  - [ ] PR-1 and PR-2 are independent and first. Both branch from `main`.
  - [ ] PR-3 after PR-2 reports, based on the PR-2 branch.
  - [ ] PR-4 after PR-3 reports, based on the PR-3 branch.
  - [ ] PR-5 after PR-4 reports, based on the PR-4 branch.
- [ ] Hold the file boundaries. Each PR touches only the files in its Files list.
- [ ] Hold the review gate. PR-2 and PR-5 wait for the operator's review in chat before merge.

### PR mechanics, for every PR

- [ ] Resolve the forge once. This repository uses `gh`. Never require `gt`.
- [ ] Open the PR ready, never draft, with `gh pr create --base <base-branch>`. A stack child targets its parent branch.
- [ ] Run the repo's lint and format check once before the PR-facing push. Push with hooks on.
- [ ] Run `/deslop` before each commit and `/no-comments` before review.
- [ ] Triage every review-bot and security-reviewer comment on its merits. Dismiss noise with a concrete reason.
- [ ] Before the merge-ready report, record the base and head SHAs.

### Verdict and merge, for every PR

- [ ] At the merge-ready head SHA, the root runs the swarm per `skills/swarm/SKILL.md`. One gates lane. The ten live lanes from the PR's **Verify, live** block. The perf lane from its **Verify, perf** block. One audit lane that reads the diff and the receipts and distrusts the PR body.
- [ ] Clean only when every lane is `PASS`. Findings go back to the owner. A new head gets a fresh swarm and a fresh verdict.
- [ ] A clean verdict appends the PR to the linear stack. The operator lands it bottom-up. No owner merges.

### Boot recipe, for every live lane

Each live lane runs in its own worktree at the PR head. This program has no UI. Every lane drives the CLI and the hooks through the `run` skill and saves its terminal transcript instead of a screenshot.

- [ ] `git fetch origin <head-branch> && git checkout <head SHA>`.
- [ ] Export `MEMORY_MCP_DB_PATH=/tmp/swarm-<pr-id>/worker-<n>/memory.db` so no lane shares a database with another.
- [ ] Deliver input only through the commands the lane names. Read-only diagnostics are `sqlite3` queries and `memory-mcp-cli status`.
- [ ] Save every command and its output to `/tmp/swarm-<pr-id>/worker-<n>/<slug>.txt` and return the paths with the report.

### Where you work

- [ ] Work only in your own worktree under `/tmp`. Create it with the exact command in your PR section. Never edit `/Users/mdenyer/VSCode/MemoryMCP` directly.
- [ ] Run every `uv` command from inside your worktree, for example `cd /tmp/<slug> && uv run pytest`.
- [ ] Never write to `~/.memory-mcp/memory.db`. For every live check set `MEMORY_MCP_DB_PATH=/tmp/<slug>-live/memory.db` in the environment first.
- [ ] Until PR-1 merges the CLI crashes on this Mac. Set `MEMORY_MCP_EMBEDDING_BACKEND=sentence-transformers` in the environment for every live check. Do not add this to any file.
- [ ] Keep a decision trail at `/tmp/<slug>/decisions.tsv` with one row per decision. Columns are `time`, `what`, `why`, `evidence`, `result`. Do not commit it. Return its path in your report.

### The TDD loop, for every behaviour change

Do these steps in this order for each numbered step in your Build list. Do not skip step 2. Do not reorder.

1. Write the test named in your PR section. Put it in the test file named there.
2. Run only that test. Paste the failing output into `decisions.tsv`. If the test passes before you change any source, your test is wrong. Rewrite it until it fails for the reason the plan describes.
3. Make the smallest source change that makes the test pass.
4. Run only that test again. It must pass.
5. Run the whole suite with `uv run pytest -p no:randomly --no-header -o addopts=""`. It must report the same failures as before your change or fewer. Record the summary line.
6. Commit with a one-line imperative message. No mention of Claude, AI, or agents anywhere in the message.

### Things you must never do

- [ ] Never add `--quiet`, `-q`, or `--silent` to any command, script, hook, or test.
- [ ] Never widen a `try` block to hide an error. If you are adding `except Exception`, the plan must have told you to.
- [ ] Never change a file outside your PR's Files list. If you believe you must, stop, write why in `decisions.tsv`, and report it. Do not do it.
- [ ] Never delete or rewrite an existing test to make it pass. If an existing test breaks because the plan told you to change behaviour, update only the assertion that encodes the old behaviour and say which one in the commit body.
- [ ] Never write a comment that narrates what the code does. Comments are only for a non-obvious why.
- [ ] Never write "Co-Authored-By", "Generated by", or any AI attribution anywhere.
- [ ] Never merge, enable auto-merge, or close a PR.

### Gates before you open the PR

Run all four. Paste the last line of each into `decisions.tsv`.

- [ ] `uv run ruff check src/ tests/` prints `All checks passed!`.
- [ ] `uv run ruff format --check src/ tests/` prints no files that would be reformatted.
- [ ] `uv run pytest -p no:randomly --no-header -o addopts=""` ends in a summary line with `0 failed` beyond the pre-existing MLX errors listed in Appendix A, and every new test named in your PR section appears as passed when run by name.
- [ ] Every command in your PR's **Verify, live** block produces the expected output written next to it.

### Opening the PR

- [ ] Push your branch with `git push -u origin <branch>`.
- [ ] Open it with `gh pr create --base <base-branch> --title "<imperative title>" --body-file /tmp/<slug>/pr-body.md`. Never draft.
- [ ] The body has one bullet per change saying what and why, one line per open risk, and the exact commands and outputs from your live block. No chronology. No attribution.
- [ ] Wait for CI with `gh pr checks <number> --watch`. If a check fails, read the log with `gh run view <run-id> --log-failed`, fix the cause, push again. After two failures with the same error, stop and report instead of retrying.
- [ ] Report the PR URL, the head SHA from `git rev-parse HEAD`, the path to `decisions.tsv`, and the unit, live, and perf results. Then stop.

## Fix the Apple Silicon crash and test the platform default (PR-1)

**Depends on.** None. Branch from `origin/main`.

**Why.** `transformers` 5.13.0 makes `import mlx_embeddings` raise `AttributeError` through `mlx_lm`. The guard `is_mlx_available()` in `src/memory_mcp/embeddings.py` line 29 only catches `ImportError`, so the server, the CLI, and every hook die at import on every Apple Silicon machine. CI runs Ubuntu only and never installs `mlx-embeddings`, so no check ever ran this path.

**Worktree.**

```bash
git -C /Users/mdenyer/VSCode/MemoryMCP fetch origin
git -C /Users/mdenyer/VSCode/MemoryMCP worktree add /tmp/fix-mlx-import -b fix-mlx-import-fallback origin/main
```

**Files.**

- [ ] Edit `src/memory_mcp/embeddings.py`.
- [ ] Edit `tests/test_embeddings.py`.
- [ ] Edit `.github/workflows/ci.yml`.
- [ ] Edit `CHANGELOG.md`.

**Build.**

- [ ] Step 1. In `is_mlx_available()` catch `Exception` instead of `ImportError`, and log one warning through the module logger that names the exception type and says the sentence-transformers backend is being used instead. This is the one place the plan allows a broad except, because any failure inside a third-party import must mean "MLX unavailable", never "crash the server".
- [ ] Step 2. Add a job named `test-macos` to `.github/workflows/ci.yml`. Copy the `test` job, set `runs-on: macos-14`, drop the matrix, use Python 3.12, and pin the same action SHAs the other jobs use. It runs `uv sync --group dev` and then `uv run pytest -p no:randomly --no-header -o addopts="" tests/test_embeddings.py`.
- [ ] Step 3. Add a CHANGELOG entry under a new `## [Unreleased]` heading, section `### Fixed`, one bullet.

**You see.**

- [ ] `uv run memory-mcp-cli status` on this Mac prints the status table instead of a traceback, and stderr contains one warning line naming `AttributeError`.

**Verify, unit.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [ ] `tests/test_embeddings.py` gains `test_is_mlx_available_false_when_import_raises_non_import_error`. It inserts a fake `mlx_embeddings.utils` module into `sys.modules` whose attribute access raises `AttributeError`, calls `is_mlx_available()`, and asserts the result is `False`. Before step 1 this test fails with `AttributeError`. Run `uv run pytest -p no:randomly --no-header -o addopts="" tests/test_embeddings.py -k non_import_error`.
- [ ] The three pre-existing errors in `TestEmbeddingEngine` (`test_engine_has_dimension`, `test_engine_caches`, `test_engine_clear_cache`) pass after step 1. Run `uv run pytest -p no:randomly --no-header -o addopts="" tests/test_embeddings.py`.

**Verify, live.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked. Ten lanes on the configured `swarm workers` model at the PR head, per the boot recipe.

- [ ] Lane 1. Regression against trunk. In `/Users/mdenyer/VSCode/MemoryMCP` run `MEMORY_MCP_DB_PATH=/tmp/fix-mlx-import-live/memory.db uv run memory-mcp-cli status 2>&1 | tail -3`. Record that trunk prints `AttributeError: 'str' object has no attribute '__module__'`. Then in the worktree run the same command. Save `lane1.txt`. Pass when the worktree prints `Hot cache is empty` and no traceback.
- [ ] Lane 2. In the worktree run `MEMORY_MCP_DB_PATH=/tmp/fix-mlx-import-live/memory.db uv run memory-mcp-cli status 2>&1 | grep -c 'AttributeError'`. Save `lane2.txt`. Pass when the count is exactly `1`, which is the warning line.
- [ ] Lane 3. In the worktree run `MEMORY_MCP_DB_PATH=/tmp/fix-mlx-import-live/memory.db timeout 20 uv run memory-mcp 2>&1 < /dev/null | head -5`. Save `lane3.txt`. Pass when the output contains no traceback. A timeout exit is expected, because the server waits on stdin.
- [ ] Lane 4. Push and watch CI. Save `ci.txt`. Pass when `gh pr checks <number>` shows `test-macos` as `pass`.
- [ ] Lane 5. Run `uv run ruff check src/ tests/`. Save `lint.txt`. Pass when `All checks passed!`.
- [ ] Lane 6. Run `uv run ruff format --check src/ tests/`. Save `format.txt`. Pass when no file is listed.
- [ ] Lane 7. Run the full suite. Save `suite.txt`. Pass when the three `TestEmbeddingEngine` errors are gone and nothing new fails.
- [ ] Lane 8. Run `uv build` then `ls dist/`. Save `build.txt`. Pass when a wheel and an sdist exist.
- [ ] Lane 9. Run `MEMORY_MCP_EMBEDDING_BACKEND=mlx MEMORY_MCP_DB_PATH=/tmp/fix-mlx-import-live/forced.db uv run memory-mcp-cli status 2>&1 | tail -3`. Save `forced-mlx.txt`. Pass when the output is either the status table or a single clear error line, never a traceback.
- [ ] Lane 10. Run `git diff origin/main --stat`. Save `diffstat.txt`. Pass when only the four files in the Files list changed.

**Verify, perf.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [ ] Metric. Wall time of `memory-mcp-cli status` on an empty DB.
- [ ] Probe. `time (MEMORY_MCP_DB_PATH=/tmp/fix-mlx-import-live/memory.db uv run memory-mcp-cli status > /dev/null 2>&1)` three times in the worktree. Trunk cannot produce the metric because it crashes, so record that and use the absolute budget.
- [ ] Baseline. Trunk crashes. Record the crash line as the baseline.
- [ ] Rule. Median under 10 seconds. Fail above that.

**Review gate.** None. PR-1 is not review-gated.

**Merge.**

- [ ] Root's clean verdict at the exact head SHA.
- [ ] CI green including `test-macos`.
- [ ] The operator merges.

## Inject the hot cache through hooks (PR-2)

**Depends on.** None. Branch from `origin/main`. Use the two environment variables from the rules for every live check until PR-1 lands.

**Why.** Claude Code adds plain stdout from `SessionStart` and `UserPromptSubmit` hooks to Claude's context. Nothing else injects. The plugin's `SessionStart` hook runs `memory-mcp-cli bootstrap --quiet`, which prints nothing. MCP resources are never auto-injected. The plugin also keeps `commands/` and `skills/` inside `.claude-plugin/`, where Claude Code does not look, so no slash command or skill loads.

**Worktree.**

```bash
git -C /Users/mdenyer/VSCode/MemoryMCP fetch origin
git -C /Users/mdenyer/VSCode/MemoryMCP worktree add /tmp/hot-cache-hook -b hot-cache-hook-injection origin/main
```

**Files.**

- [ ] Edit `src/memory_mcp/storage/core.py`.
- [ ] Edit `src/memory_mcp/helpers.py`.
- [ ] Edit `src/memory_mcp/cli.py`.
- [ ] Edit `.claude-plugin/plugin.json`.
- [ ] Move `.claude-plugin/commands/` to `commands/`.
- [ ] Move `.claude-plugin/skills/recall-nudge/` to `skills/recall-nudge/`.
- [ ] Move `.claude-plugin/resources/` to `commands/resources/` and fix every relative link inside `commands/*.md` that pointed at it.
- [ ] Edit `tests/test_plugin.py`.
- [ ] Edit `tests/test_storage.py`.
- [ ] Edit `tests/test_cli.py`.
- [ ] Edit `README.md`, `CLAUDE.md`, `CHANGELOG.md`.

**Build.**

- [ ] Step 1. Make the embedding engine lazy. In `Storage.__init__` in `src/memory_mcp/storage/core.py` replace `self._embedding_engine = EmbeddingEngine(self.settings)` with `self._embedding_engine_instance = None`, and add a property `_embedding_engine` that builds `EmbeddingEngine(self.settings)` on first access and caches it. Every existing use of `self._embedding_engine` keeps working. A hook that only reads SQL must never load a model.
- [ ] Step 2. Add `format_hot_cache_for_injection(memories, max_chars)` to `src/memory_mcp/helpers.py`. It returns an empty string for an empty list. Otherwise it returns a header line `[MEMORY: Hot cache]`, then one line per memory in the form `- [id:<id>] <content truncated to max_chars with ... when cut> [<up to three tags comma separated>]`, then a final line `Call mark_memory_used(id) when one of these was useful.` It is a pure function with no storage access.
- [ ] Step 3. Add a CLI command `hot-cache` in `src/memory_mcp/cli.py` with one flag `--force`. It does these things in this order. Read all of stdin if stdin is not a TTY, parse it as JSON if it is non-empty, and take `session_id` from it; a missing or unparsable stdin means `session_id` is `None`. Build `Storage(settings)`. Call `storage.get_hot_cache()`. If the list is empty, print nothing and exit 0. Format it with `format_hot_cache_for_injection` using `settings.hot_cache_display_max_chars`. Compute the SHA-256 of the formatted text. If `session_id` is set and `--force` is not given, read `~/.memory-mcp/injected/<session_id>` and if it holds the same hash, print nothing and exit 0. Otherwise print the text, write the hash to that file (create the directory), call `storage.log_injections_batch(memory_ids, resource="hook", session_id=session_id, project_id=get_current_project_id())`, and exit 0. Any exception inside the command is written to stderr and the command exits 0, because a broken hook must never block Claude. The `~/.memory-mcp` directory comes from `settings.db_path.parent`, never a hard-coded string.
- [ ] Step 4. Edit `.claude-plugin/plugin.json`. Replace the `SessionStart` command with `memory-mcp-cli hot-cache --force`. Add a `UserPromptSubmit` entry with matcher `""` and command `memory-mcp-cli hot-cache`. Leave `Stop` and `PreCompact` exactly as they are. Do not add `bootstrap` anywhere.
- [ ] Step 5. Move the directories listed in Files with `git mv`. Then run `claude plugin validate /tmp/hot-cache-hook` and fix everything it prints until it reports success.
- [ ] Step 6. Update `tests/test_plugin.py` paths to the new locations, and add a test that asserts `.claude-plugin/` contains only `plugin.json` and `marketplace.json`.
- [ ] Step 7. Update `README.md` and `CLAUDE.md` so every sentence that says the hot cache is auto-injected names the hooks as the mechanism. Add a CHANGELOG entry under `## [Unreleased]`, sections `### Added` and `### Changed`.

**You see.**

- [ ] `echo '{"session_id":"s1"}' | memory-mcp-cli hot-cache` on a DB with two promoted memories prints a header, two `[id:N]` lines, and the mark-used sentence. Run again with the same session id and it prints nothing. Run with `--force` and it prints again.

**Verify, unit.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [ ] `tests/test_storage.py` gains `test_storage_does_not_build_embedding_engine_until_needed`. It patches `memory_mcp.storage.core.EmbeddingEngine` with a `Mock(side_effect=RuntimeError("must not load"))`, constructs `Storage(temp_settings)`, calls `get_hot_cache()` and asserts it returns a list, then calls `store_memory("x")` and asserts `RuntimeError` is raised. Before step 1 the constructor raises. Run `uv run pytest -p no:randomly --no-header -o addopts="" tests/test_storage.py -k until_needed`.
- [ ] `tests/test_server.py` or a new `tests/test_helpers.py` gains `test_format_hot_cache_for_injection_shape`. It builds two `Memory` objects and asserts the exact output string, and asserts the empty list returns `""`. Run by name.
- [ ] `tests/test_cli.py` gains a class `TestHotCacheCommand` with these tests. `test_prints_nothing_when_empty`. `test_prints_memories_with_ids` after seeding two memories and promoting them through `Storage`. `test_second_call_same_session_prints_nothing`. `test_force_prints_again`. `test_logs_injection_rows_with_resource_hook`, which asserts `injection_log` has two rows with `resource='hook'`. `test_exception_goes_to_stderr_and_exit_zero`, which patches `Storage.get_hot_cache` to raise and asserts exit code 0 and empty stdout. Use `patch("sys.stdin", io.StringIO('{"session_id":"s1"}'))` and `capsys`. Run `uv run pytest -p no:randomly --no-header -o addopts="" tests/test_cli.py -k TestHotCacheCommand`.
- [ ] `tests/test_plugin.py` gains `test_plugin_components_live_at_root` and `test_plugin_hooks_inject_hot_cache`, the second asserting `SessionStart` runs `memory-mcp-cli hot-cache --force` and `UserPromptSubmit` runs `memory-mcp-cli hot-cache`.

**Verify, live.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked. Ten lanes on the configured `swarm workers` model at the PR head, per the boot recipe. Every lane runs in `/tmp/hot-cache-hook` with `export MEMORY_MCP_DB_PATH=/tmp/hot-cache-hook-live/memory.db` and `export MEMORY_MCP_EMBEDDING_BACKEND=sentence-transformers`.

- [ ] Lane 1. Regression against trunk. Trunk has no `hot-cache` command. Record `uv run memory-mcp-cli hot-cache` on trunk printing `No such command`. Then gate the diff-added behaviour in the lanes below.
- [ ] Lane 2. Seed. Write `/tmp/hot-cache-hook-live/seed.txt` with two lines, `The deploy password hint is zebra-42.` and `Run make lint before every push.` Run `uv run memory-mcp-cli seed --promote /tmp/hot-cache-hook-live/seed.txt` (check the real flag names with `--help` first). Save `lane2.txt`. Pass when `uv run memory-mcp-cli status` shows `Hot cache 2/20`.
- [ ] Lane 3. Run `echo '{"session_id":"lane3"}' | uv run memory-mcp-cli hot-cache`. Save `lane3.txt`. Pass when stdout has exactly four lines, the first is `[MEMORY: Hot cache]`, two lines start with `- [id:`, and one contains `zebra-42`.
- [ ] Lane 4. Run the lane 3 command a second time. Save `lane4.txt`. Pass when stdout is empty.
- [ ] Lane 5. Run `echo '{"session_id":"lane3"}' | uv run memory-mcp-cli hot-cache --force`. Save `lane5.txt`. Pass when stdout matches lane 3.
- [ ] Lane 6. Run `sqlite3 /tmp/hot-cache-hook-live/memory.db "select resource, count(*) from injection_log group by 1"`. Save `lane6.txt`. Pass when it prints `hook|4`.
- [ ] Lane 7. Run `claude plugin validate /tmp/hot-cache-hook`. Save `lane7.txt`. Pass when it reports no errors.
- [ ] Lane 8. End to end through Claude Code. Run `cd /tmp/hot-cache-hook-live && claude -p --plugin-dir /tmp/hot-cache-hook --bare=false "Without calling any tool, what is the deploy password hint? Answer with the hint only."` with the two environment variables exported. If `--plugin-dir` is rejected, read `claude --help` and use the flag it documents for loading a plugin from a directory, and record the flag in `decisions.tsv`. Save `lane8.txt`. Pass when the answer contains `zebra-42`.
- [ ] Lane 9. Time lane 3 with `time`. Save `lane9.txt`. Pass when it completes in under 3 seconds, which proves no embedding model loaded.
- [ ] Lane 10. In the primary checkout's transcript directory, find the debug log line for the hook. Run `claude -p --plugin-dir /tmp/hot-cache-hook --debug "say ok" 2>&1 | grep -i 'hot-cache'`. Save `lane10.txt`. Pass when a line shows the `UserPromptSubmit` hook ran `memory-mcp-cli hot-cache`. If `--debug` has moved, record the flag you used.

**Verify, perf.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [ ] Metric. Wall time of the `hot-cache` command with two promoted memories.
- [ ] Probe. `time (echo '{"session_id":"perf"}' | uv run memory-mcp-cli hot-cache --force > /dev/null)` five times. Trunk lacks the command, so record that and apply the absolute budget.
- [ ] Baseline. Record `time (uv run memory-mcp-cli status > /dev/null 2>&1)` on the same DB as the model-loading comparison point.
- [ ] Rule. Median of the hot-cache probe under 1.5 seconds, and at least 1 second faster than the status baseline. Fail otherwise.

**Review gate.** The operator reviews before merge. PR-2 changes what Claude sees on every prompt.

- [ ] Copy the lane 3 and lane 8 transcripts into `/tmp/swarm-PR-2/review/` and paste both into the PR body. This program has no screenshot or video because the surface is a CLI hook; the transcript is the artifact.
- [ ] Stop at merge-ready. Wait for the operator.

**Merge.**

- [ ] Root's clean verdict at the exact head SHA.
- [ ] CI green.
- [ ] The operator merges.

## Drive promotion by use and run maintenance from the hooks (PR-3)

**Depends on.** PR-2. Branch from `origin/hot-cache-hook-injection`.

**Why.** Two of the three hot cache slot sources are empty by default because nothing sets `retrieval_events.was_used`. The Stop hook already has a token matcher that finds injected memories in Claude's reply, but it only bumps `used_count`. The same hook crashes on a third of real transcripts because it assumes `message.content` is a list. Auto-demote only runs inside two manual MCP tools. And the v0.7 rename left the promoted set capped by the wrong setting.

**Worktree.**

```bash
git -C /Users/mdenyer/VSCode/MemoryMCP fetch origin
git -C /Users/mdenyer/VSCode/MemoryMCP worktree add /tmp/promote-by-use -b promote-by-use origin/hot-cache-hook-injection
```

**Files.**

- [ ] Edit `src/memory_mcp/cli.py`.
- [ ] Edit `src/memory_mcp/storage/injection_tracking.py`.
- [ ] Edit `src/memory_mcp/storage/hot_cache.py`.
- [ ] Edit `src/memory_mcp/storage/memory_crud.py`.
- [ ] Edit `src/memory_mcp/storage/retrieval.py`.
- [ ] Edit `src/memory_mcp/server/app.py`.
- [ ] Edit `tests/test_cli.py`, `tests/test_storage.py`, `tests/test_bug_regressions.py`.
- [ ] Edit `CHANGELOG.md`.

**Build.**

- [ ] Step 1. Fix the transcript parser in `log_response` in `src/memory_mcp/cli.py`. Add a module-level function `_text_of_content(content)` that returns the string itself when `content` is a `str`, joins the `text` fields of `type == "text"` dicts when it is a list, and returns `""` for anything else. Use it in the loop. Nothing else in the loop changes.
- [ ] Step 2. In `mark_used_memories` in `src/memory_mcp/storage/injection_tracking.py`, for every memory it marks, also call `self.mark_retrieval_used(...)` or insert a `retrieval_events` row with `was_used = 1` for that memory and session, whichever the existing `mark_retrieval_used` signature at `src/memory_mcp/storage/retrieval.py` line 71 supports. Read that function first. The outcome that matters is that `get_recent_recalls()` returns the memory afterwards.
- [ ] Step 3. At the end of `log_response`, after `mark_used_memories`, call `storage.demote_stale_hot_memories()` and `storage.improve_hot_cache_from_injections(dry_run=False)`. Wrap the pair in one `try` that logs the exception to stderr and continues. This is the second and last broad except the plan allows, for the same reason as the hook command.
- [ ] Step 4. In `promote_to_hot` in `src/memory_mcp/storage/hot_cache.py` line 246, replace `settings.hot_cache_max_items` with `settings.promoted_max_items`.
- [ ] Step 5. In `_compute_salience_score` in `src/memory_mcp/storage/memory_crud.py` line 636, replace `self.settings.hot_cache_max_items` with `self.settings.promoted_max_items`.
- [ ] Step 6. Give `get_hot_cache` in `src/memory_mcp/storage/retrieval.py` a `project_id: str | None = None` parameter and pass it through to `get_promoted_memories(project_id=project_id)` inside `_get_promoted_by_salience`. Pass `get_current_project_id()` from the `hot-cache` CLI command and from `hot_cache_resource` in `src/memory_mcp/server/app.py`.
- [ ] Step 7. In `promoted_memories_resource` in `src/memory_mcp/server/app.py` line 281 change `resource="hot-cache"` to `resource="promoted-memories"`. In `hot_cache_resource` call `storage.record_hot_cache_hit()` when the list is non-empty and `storage.record_hot_cache_miss()` when it is empty.
- [ ] Step 8. CHANGELOG entries under `## [Unreleased]`, `### Fixed`.

**You see.**

- [ ] A Stop hook run over a transcript whose user turn has string content exits 0 and logs the output. A Stop hook run whose assistant text mentions `zebra-42` leaves that memory in `get_recent_recalls()` and first in the next `hot-cache --force` output.

**Verify, unit.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [ ] `tests/test_bug_regressions.py` gains `test_log_response_accepts_string_user_content`. It writes a two-line JSONL transcript where the user line has `"content": "What is X?"` as a string and the assistant line has list content, feeds `{"session_id": "s", "transcript_path": <path>}` on stdin, runs `main()` with argv `["memory-mcp-cli", "log-response"]`, and asserts exit 0 and one `output_log` row. Before step 1 this raises `AttributeError: 'str' object has no attribute 'get'`.
- [ ] `tests/test_storage.py` gains `test_mark_used_memories_populates_recent_recalls`. Seed a memory containing the token `zebra-42`, log an injection for it with `resource="hook"`, call `mark_used_memories("the hint is zebra-42")`, assert `get_recent_recalls()` contains the memory id.
- [ ] `tests/test_cli.py` gains `test_log_response_demotes_stale_hot_memory`. Promote a memory, set its `last_accessed_at` 30 days back with a direct SQL update, run `log-response` with a minimal transcript, assert `is_hot` is 0 afterwards.
- [ ] `tests/test_storage.py` gains `test_promote_to_hot_caps_at_promoted_max_items`. With `promoted_max_items=20` and `hot_cache_max_items=10`, promote 15 memories and assert 15 are hot. Before step 4 only 10 are hot.
- [ ] `tests/test_storage.py` gains `test_get_hot_cache_filters_by_project`. Two promoted memories in different projects; `get_hot_cache(project_id="a")` returns only project a plus globals.
- [ ] `tests/test_server.py` gains `test_promoted_resource_logs_distinct_resource_name` asserting the `injection_log` row says `promoted-memories`.

**Verify, live.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked. Ten lanes on the configured `swarm workers` model at the PR head, per the boot recipe. Every lane runs in `/tmp/promote-by-use` with `export MEMORY_MCP_DB_PATH=/tmp/promote-by-use-live/memory.db` and the sentence-transformers variable.

- [ ] Lane 1. Regression against trunk. On the PR-2 branch, run `log-response` over `/tmp/promote-by-use-live/string-content.jsonl`, a transcript with a string user turn. Record the `AttributeError`. Run the same on the PR-3 head. Save `lane1.txt`. Pass when exit is 0 and `sqlite3 ... "select count(*) from output_log"` prints `1`.
- [ ] Lane 2. Seed and promote the two lane-2 memories from PR-2. Run `hot-cache --force` and record the order.
- [ ] Lane 3. Write `/tmp/promote-by-use-live/used.jsonl` whose last assistant text is `The hint is zebra-42, as I recall.` Pipe `{"session_id":"lane3","transcript_path":"/tmp/promote-by-use-live/used.jsonl"}` into `log-response`. Save `lane3.txt`. Pass when `sqlite3 ... "select count(*) from retrieval_events where was_used=1"` prints `1`.
- [ ] Lane 4. Run `echo '{"session_id":"lane4"}' | uv run memory-mcp-cli hot-cache --force`. Save `lane4.txt`. Pass when the `zebra-42` line is the first `[id:` line.
- [ ] Lane 5. Run `sqlite3 ... "update memories set last_accessed_at = datetime('now','-40 days') where content like '%make lint%'"`, then pipe the lane 3 input into `log-response` again. Save `lane5.txt`. Pass when `sqlite3 ... "select is_hot from memories where content like '%make lint%'"` prints `0`.
- [ ] Lane 6. Promote 15 memories through `seed` and `promote`. Save `lane6.txt`. Pass when `status` shows `Hot cache 15/20`.
- [ ] Lane 7. Run the full PR-2 lane 8 end-to-end Claude Code check again on this head. Save `lane7.txt`. Pass when the answer contains `zebra-42`.
- [ ] Lane 8. Run the full suite. Save `lane8.txt`. Pass when the summary shows no new failures.
- [ ] Lane 9. Run `time` on lane 3. Save `lane9.txt`. Pass when under 5 seconds, because `log-response` may load the model for `improve_hot_cache_from_injections`. Record the time.
- [ ] Lane 10. Run `uv run memory-mcp-cli status`. Save `lane10.txt`. Pass when `Learning Loop` still renders without error.

**Verify, perf.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [ ] Metric. Wall time of `log-response` over `used.jsonl`.
- [ ] Probe. `time` it five times on the PR-2 head and five times on the PR-3 head, interleaved.
- [ ] Baseline. Record the PR-2 median first.
- [ ] Rule. PR-3 median no more than 2 seconds slower than PR-2. Fail otherwise.

**Review gate.** None. PR-3 is not review-gated.

**Merge.**

- [ ] Root's clean verdict at the exact head SHA.
- [ ] CI green.
- [ ] The operator merges after PR-2.

## Delete mining and its observability (PR-4)

**Depends on.** PR-3. Branch from `origin/promote-by-use`.

**Why.** Mining has produced zero patterns in every recorded run. It has no working confidence gate, it promotes to the hot cache without the promotion gates, it inflates its own access counts through `recall()`, and everything in v0.8 (probe, `mining_runs`, staleness warning, `hook-check`) exists to watch it. PR-3 replaced its purpose with promotion by use. The archived branch `archive/simplify-and-analysis` at `b502f72` did this deletion once and is a shape reference only. Do not merge or cherry-pick from it.

**Worktree.**

```bash
git -C /Users/mdenyer/VSCode/MemoryMCP fetch origin
git -C /Users/mdenyer/VSCode/MemoryMCP worktree add /tmp/remove-mining -b remove-mining origin/promote-by-use
```

**Files.**

- [ ] Delete `src/memory_mcp/mining.py`, `src/memory_mcp/probe.py`, `src/memory_mcp/storage/mining_store.py`, `src/memory_mcp/storage/mining_runs.py`, `src/memory_mcp/storage/output_logging.py`, `src/memory_mcp/server/tools/mining.py`, `src/memory_mcp/dashboard/templates/mining.html`, `hooks/memory-log-response.sh`, `tests/test_mining.py`, `tests/test_loop_observability.py`, `tests/test_hook.py`, `commands/mining.md`, `commands/resources/testing/MINING.md`, `docs/examples/pattern-mining.md`.
- [ ] Edit `src/memory_mcp/cli.py` to remove `log-output`, `run-mining`, `hook-check`, and `_loop_warning_line`, and to strip mining from `log-response` and `pre-compact`.
- [ ] Edit `src/memory_mcp/storage/core.py`, `src/memory_mcp/storage/__init__.py`, `src/memory_mcp/server/__init__.py`, `src/memory_mcp/server/tools/__init__.py`, `src/memory_mcp/server/app.py`, `src/memory_mcp/config.py`, `src/memory_mcp/migrations.py`, `src/memory_mcp/dashboard/app.py`, `src/memory_mcp/dashboard/templates/base.html`, `src/memory_mcp/storage/maintenance.py`, `src/memory_mcp/helpers.py`, `src/memory_mcp/ml_classification.py`.
- [ ] Edit `tests/test_cli.py`, `tests/test_storage.py`, `tests/test_server.py`, `tests/test_dashboard.py`, `tests/test_bug_regressions.py` to remove tests of deleted code only.
- [ ] Edit `README.md`, `CLAUDE.md`, `CHANGELOG.md`, `docs/API.md`, `docs/REFERENCE.md`, `docs/CODEMAP.md`, `docs/TROUBLESHOOTING.md`, `ARCHITECTURE.md`, `CONTRIBUTING.md`, `.claude-plugin/plugin.json`.

**Build.**

- [ ] Step 1. Write the migration first. Add `migrate_v18_to_v19` in `src/memory_mcp/migrations.py` that drops `mining_runs`, `mined_patterns`, and `output_log`, and bump `SCHEMA_VERSION` to 19. Write its test before the code.
- [ ] Step 2. Delete the files in the Files list with `git rm`. Run `uv run ruff check src/ tests/` and fix every unresolved import it reports by removing the import and the code that used it. Repeat until ruff is clean. Do not stub anything.
- [ ] Step 3. `log_response` keeps transcript parsing, `log_output` is gone, so replace the `storage.log_output(...)` call with nothing. Keep `mark_used_memories`, `demote_stale_hot_memories`, and `improve_hot_cache_from_injections`. Delete the `subprocess.Popen` that spawned mining.
- [ ] Step 4. `pre_compact` keeps `end_session()` and drops the mining spawn and the `--skip-mining` flag.
- [ ] Step 5. Remove every `mining_*`, `ner_*`, `log_retention_days`, `loop_warnings_enabled`, `warn_missing_hook`, and `max_content_length` field from `Settings` in `src/memory_mcp/config.py`, and `check_stop_hook_configured` and `get_hook_install_instructions` if nothing else uses them. Grep each name across `src/` before deleting it.
- [ ] Step 6. Remove the mining page, its route, its nav link, and its partials from the dashboard.
- [ ] Step 7. Remove `MiningRunsMixin`, `MiningStoreMixin`, and `OutputLoggingMixin` from `Storage` and its docstring.
- [ ] Step 8. Remove NER. Delete `extract_entities_ner` callers and any `transformers` pipeline import that only mining used. Keep `transformers` in `pyproject.toml` because `sentence-transformers` needs it.
- [ ] Step 9. Change `bootstrap` so it never promotes by default (`--promote` becomes opt-in) and skips `CLAUDE.md` and `.claude/CLAUDE.md`, which Claude Code already injects. Edit `BOOTSTRAP_DEFAULT_FILES` in `src/memory_mcp/config.py`.
- [ ] Step 10. Update every doc so no sentence mentions mining, pattern extraction, the Stop hook loop, `hook-check`, `run-mining`, `log-output`, or the Memory Analyst agent. Keep the CHANGELOG history untouched below `## [Unreleased]` and add a `### Removed` section above it.

**You see.**

- [ ] `grep -rn -i 'mining\|mined_pattern\|output_log' src/ tests/ commands/ docs/ README.md CLAUDE.md` prints nothing except CHANGELOG history.

**Verify, unit.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [ ] `tests/test_storage.py` gains `test_migration_v19_drops_mining_tables`. Build a v18 DB by constructing `Storage` on the PR-3 head schema in a fixture (or by creating the three tables by hand at v18), open it on the PR-4 head, and assert `sqlite_master` has none of the three tables and `schema_version` is 19.
- [ ] `tests/test_cli.py` gains `test_bootstrap_skips_claude_md_and_does_not_promote`. A temp root with `CLAUDE.md` and `README.md`; after `bootstrap`, no memory content comes from `CLAUDE.md` and `is_hot` is 0 for all.
- [ ] `tests/test_cli.py` gains `test_log_response_still_marks_used_without_mining`, which is the PR-3 lane 3 scenario as a unit test.
- [ ] Full suite green. Record the new test count; it must be lower than the PR-3 count by at least 190, the number of tests in the deleted files.

**Verify, live.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked. Ten lanes on the configured `swarm workers` model at the PR head, per the boot recipe. Every lane runs in `/tmp/remove-mining` with `export MEMORY_MCP_DB_PATH=/tmp/remove-mining-live/memory.db`.

- [ ] Lane 1. Regression against trunk. Copy the PR-3 live DB to `/tmp/remove-mining-live/memory.db`. Run `status` on the PR-3 head and record the table. Run `status` on the PR-4 head. Save `lane1.txt`. Pass when the memory and hot cache counts match and no `Learning Loop` section appears.
- [ ] Lane 2. Run `sqlite3 /tmp/remove-mining-live/memory.db "select name from sqlite_master where name in ('mining_runs','mined_patterns','output_log')"` after lane 1. Save `lane2.txt`. Pass when empty.
- [ ] Lane 3. Pipe the PR-3 `used.jsonl` input into `log-response`. Save `lane3.txt`. Pass when exit 0 and `retrieval_events` gains a `was_used=1` row.
- [ ] Lane 4. Run `uv run memory-mcp-cli --help`. Save `lane4.txt`. Pass when `log-output`, `run-mining`, and `hook-check` are absent.
- [ ] Lane 5. Run `uv run memory-mcp-cli dashboard --port 8799 &` then `curl -s localhost:8799/ | grep -c -i mining`. Save `lane5.txt`. Pass when `0`. Kill the server.
- [ ] Lane 6. Run `curl -s localhost:8799/mining -o /dev/null -w '%{http_code}'`. Save `lane6.txt`. Pass when `404`.
- [ ] Lane 7. Run `claude plugin validate /tmp/remove-mining`. Save `lane7.txt`. Pass when clean.
- [ ] Lane 8. Run the PR-2 lane 8 end-to-end Claude Code check. Save `lane8.txt`. Pass when the answer contains `zebra-42`.
- [ ] Lane 9. Run `uv run python -c "import memory_mcp.server"`. Save `lane9.txt`. Pass when it prints nothing and exits 0.
- [ ] Lane 10. Run `wc -l src/memory_mcp/*.py src/memory_mcp/**/*.py | tail -1` on PR-3 and PR-4 heads. Save `lane10.txt`. Pass when PR-4 is at least 2,500 lines smaller.

**Verify, perf.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [ ] Metric. Wall time of the full test suite.
- [ ] Probe. `time uv run pytest -p no:randomly --no-header -o addopts=""` on PR-3 and PR-4 heads.
- [ ] Baseline. PR-3 first.
- [ ] Rule. PR-4 faster or equal. Fail if slower.

**Review gate.** None. PR-4 is not review-gated.

**Merge.**

- [ ] Root's clean verdict at the exact head SHA.
- [ ] CI green.
- [ ] The operator merges after PR-3.

## Trim the tool surface and drop the MLX dependency chain (PR-5)

**Depends on.** PR-4. Branch from `origin/remove-mining`. The operator may drop this PR without affecting the others.

**Why.** 56 MCP tools remain after mining is gone, and 24 are reachable only if Claude picks them unprompted. `mlx-embeddings` pulls `mlx-vlm`, `mlx-audio`, `mlx-lm`, `opencv`, and `datasets` for a 384-dimension MiniLM model, and it is the chain that broke in PR-1.

**Worktree.**

```bash
git -C /Users/mdenyer/VSCode/MemoryMCP fetch origin
git -C /Users/mdenyer/VSCode/MemoryMCP worktree add /tmp/trim-surface -b trim-surface origin/remove-mining
```

**Files.**

- [ ] Edit `pyproject.toml`, `uv.lock`, `src/memory_mcp/embeddings.py`, `tests/test_embeddings.py`, `.github/workflows/ci.yml`.
- [ ] Edit every file under `src/memory_mcp/server/tools/`, `src/memory_mcp/server/tools/__init__.py`, `tests/test_server.py`, `docs/API.md`, `README.md`, `CLAUDE.md`, `CHANGELOG.md`, and the `commands/*.md` that reference removed tools.

**Build.**

- [ ] Step 1. Remove the `mlx-embeddings` dependency line and the `mlx` optional extra from `pyproject.toml`. Delete `MLXEmbeddingProvider`, `is_mlx_available`, `_should_use_mlx`, and the `embedding_backend` values `auto` and `mlx` from `src/memory_mcp/embeddings.py`, so `sentence-transformers` is the only provider. Run `uv lock` and commit `uv.lock`. Keep the `test-macos` CI job from PR-1; it now proves the single provider works on Apple Silicon.
- [ ] Step 2. Keep exactly the sixteen MCP tools listed here and remove every other `@mcp.tool`. Keep `remember`, `recall`, `forget`, `list_memories`, `memory_stats`, `hot_cache_status`, `promote`, `demote`, `pin`, `unpin`, `mark_memory_used`, `link_memories`, `get_related_memories`, `end_session`, `bootstrap_project`, `db_maintenance`. Storage methods that the CLI or the dashboard still call stay. Storage methods with no remaining caller go. Grep before deleting each one.
- [ ] Step 3. Update `docs/API.md` to list only the kept tools, and every `commands/*.md` that referenced a removed tool to reference a kept one or be deleted.
- [ ] Step 4. CHANGELOG `### Removed` and `### Changed` bullets.

**You see.**

- [ ] `uv run python -c "from memory_mcp.server.app import mcp; import asyncio; print(len(asyncio.run(mcp.get_tools())))"` prints `16`. If `get_tools` is named differently in this FastMCP version, use the method `fastmcp` documents and record it.

**Verify, unit.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [ ] `tests/test_server.py` gains `test_registered_tool_names_are_exactly_the_kept_set`, asserting the set of registered tool names equals the 16 above.
- [ ] `tests/test_embeddings.py` loses its MLX tests and gains `test_create_provider_returns_sentence_transformers_on_every_platform`.
- [ ] Full suite green.

**Verify, live.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked. Ten lanes on the configured `swarm workers` model at the PR head, per the boot recipe. Every lane runs in `/tmp/trim-surface` with `export MEMORY_MCP_DB_PATH=/tmp/trim-surface-live/memory.db` and without the sentence-transformers variable, because it no longer exists.

- [ ] Lane 1. Regression against trunk. `uv tree --depth 1 | wc -l` on PR-4 and PR-5 heads. Record both. Save `lane1.txt`. Pass when PR-5 is smaller and `uv tree | grep -c mlx` prints `0`.
- [ ] Lane 2. Run `uv run memory-mcp-cli status` with no environment override on this Mac. Save `lane2.txt`. Pass when it prints the table and no warning about MLX.
- [ ] Lane 3. Run the tool-count command from **You see**. Save `lane3.txt`. Pass when `16`.
- [ ] Lane 4. Run the PR-2 lane 8 end-to-end Claude Code check with no environment override. Save `lane4.txt`. Pass when the answer contains `zebra-42`.
- [ ] Lane 5. In a Claude Code session with the plugin loaded, run `/memory-mcp:remember the build tool is uv` and then `/memory-mcp:recall build tool`. Save `lane5.txt`. Pass when recall returns the memory.
- [ ] Lane 6. Run `claude plugin validate /tmp/trim-surface`. Save `lane6.txt`. Pass when clean.
- [ ] Lane 7. Run `du -sh .venv` on PR-4 and PR-5 heads after `uv sync`. Record both. Save `lane7.txt`. Pass when PR-5 is smaller.
- [ ] Lane 8. Run `grep -rn 'mlx' src/ tests/ pyproject.toml`. Save `lane8.txt`. Pass when empty.
- [ ] Lane 9. Run `uv run memory-mcp-cli dashboard --port 8798 &` and `curl -s -o /dev/null -w '%{http_code}' localhost:8798/`. Save `lane9.txt`. Pass when `200`. Kill the server.
- [ ] Lane 10. Full suite. Save `lane10.txt`. Pass when green.

**Verify, perf.** Tests alone are not sufficient verification. A PR is verified only when its unit, live, and perf boxes are all checked.

- [ ] Metric. Wall time of `uv sync --reinstall` from a clean cache.
- [ ] Probe. Once on PR-4 and once on PR-5, `UV_CACHE_DIR=/tmp/trim-surface-cache uv sync --reinstall`.
- [ ] Baseline. PR-4 first.
- [ ] Rule. PR-5 faster. Fail if slower.

**Review gate.** The operator reviews before merge. PR-5 removes user-visible tools.

- [ ] Copy the lane 3 and lane 5 transcripts into `/tmp/swarm-PR-5/review/` and paste the removed tool list into the PR body. This program has no screenshot or video because the surface is a CLI and MCP tool list; the transcript is the artifact.
- [ ] Stop at merge-ready. Wait for the operator.

**Merge.**

- [ ] Root's clean verdict at the exact head SHA.
- [ ] CI green.
- [ ] The operator merges after PR-4, or drops this PR.

## Close the program

- [ ] Every box above is checked with its evidence.
- [ ] Reply to the operator with the stack root and tip, one verdict line per PR, and anything parked with its reason.

## Appendix A. Prototype evidence

Evidence gathered on 2026-09-11 by running the real artifacts. No prototype branch was needed; every open question was settled by reading docs or running the CLI.

Pre-existing test errors on `main` at `f81b0ba` on this Mac. `tests/test_embeddings.py::TestEmbeddingEngine::test_engine_has_dimension`, `test_engine_caches`, `test_engine_clear_cache`, all raising `AttributeError: 'str' object has no attribute '__module__'` from `transformers/models/auto/auto_factory.py:680`. Full suite otherwise 736 passed, 2 skipped.

Claude Code hooks documentation, "Exit code 0" section, states that `UserPromptSubmit`, `UserPromptExpansion`, `SessionStart`, and `PostModelSwitch` add plain-text stdout as context. The MCP documentation states resources are reachable only through `@` mentions or the resource tools. The plugins reference states components must sit at the plugin root, not inside `.claude-plugin/`.

Live database `~/.memory-mcp/memory.db` at schema v18 holds 0 memories, 0 sessions, 0 injections, 5 `mining_runs` from 2026-07-08 with 0 patterns. The plugin is not installed on this machine.

A scratch run of the mining pipeline over one realistic response produced 8 memories on the first run and 10 hot promotions by the third run of the same log row, with no user interaction. The `log-response` transcript parser raised `AttributeError` on 886 of 2,696 transcripts under `~/.claude/projects`.

## Appendix B. Alternatives rejected

Pin `transformers` below 5.13 to keep MLX working. Rejected for PR-1 because it holds every user on an old line to protect a transitive dependency that has no upper bound of its own. PR-5 removes the chain instead.

Keep mining with a working confidence gate and a processed-log marker. Rejected because the archived branch shows the project already judged the loop not worth keeping once, and promotion by use covers the same goal with code that already exists.

Print the hot cache on every prompt regardless of change. Rejected because identical text on every turn bloats context. The per-session hash file makes the injection idempotent.

Make `UserPromptSubmit` the only hook. Rejected because `SessionStart` with `--force` is needed after `clear` and `compact`, where the context is gone but the hash file says it was already injected.

## Appendix C. Risks

PR-2. `claude --plugin-dir` may not exist in this Claude Code version. The owner reads `claude --help` and records the substitute. If no flag loads a plugin from a directory, the owner installs from the worktree path with `claude plugin add` and uninstalls afterwards.

PR-2. The `UserPromptSubmit` hook spawns a Python process on every prompt. The perf rule of 1.5 seconds median holds only if the embedding engine stays lazy. Any later change that touches `Storage.__init__` must keep the unit test `test_storage_does_not_build_embedding_engine_until_needed` green.

PR-3. `improve_hot_cache_from_injections(dry_run=False)` may promote through `recall`, which loads the model inside the Stop hook. If lane 9 exceeds 5 seconds, the owner removes that call from `log_response` and records the decision. Demotion stays.

PR-4. Deleting `output_log` loses the transcript-derived text that `mark_used_memories` reads. Check that function reads `injection_log` and `memories`, not `output_log`, before deleting the table. If it reads `output_log`, keep the table and only delete the mining tables.

PR-5. Removing 40 tools breaks any user script that calls them. The operator can drop the PR. The CHANGELOG names every removed tool.

## Appendix D. Reading list

Before editing, each owner reads `CLAUDE.md` at the repo root, `docs/CODEMAP.md`, and the file list in its own section. PR-2 and PR-3 owners read `src/memory_mcp/server/app.py` lines 179 to 350 to see the existing formatting and the resource bodies. PR-4 owner reads `git show b502f72 --stat` for the shape of the earlier deletion. Every owner keeps `/tmp/<slug>/decisions.tsv`.
