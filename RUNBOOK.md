# How to run this — plan with Opus, execute with Sonnet, loop automatically

## First, one correction

There is **no `opusplan` mode** in Claude Code. Nothing automatically plans with Opus and
then switches itself to Sonnet for execution. I checked the current documentation and it
does not exist, so anything that tells you to set `model: opusplan` will silently do
nothing.

What *does* exist gets you the same outcome in two commands, because planning and
execution are already two separate sessions.

## Setup, once

```bash
mkdir clause && cd clause && git init
mkdir -p .claude/hooks
# copy PROMPT.md, CLAUDE.md, STATUS.md into the repo root
# copy .claude/settings.local.json and .claude/hooks/gate.sh into .claude/
chmod +x .claude/hooks/gate.sh
printf '.claude/settings.local.json\n.claude/.gate-count\n' >> .gitignore
```

`settings.local.json` is the machine-local settings file and Claude Code treats it as
gitignored — keep it that way so your permission grants never reach the repo.

## Step 1 — plan, with Opus

```bash
claude --model opus --permission-mode plan
```

Then paste the contents of `PROMPT.md` as your first message.

Plan mode means Claude reads, researches and proposes, but cannot write a file or run a
mutating command until you approve. Opus is the right model here because the expensive
mistakes in this project are design mistakes — the chunking strategy, the citation
schema, the golden-set design. Those are worth the stronger model.

`Shift+Tab` cycles permission modes if you want to switch mid-session.

Read the plan properly. Push back on it. Approve only when it is right.

## Step 2 — execute, with Sonnet

```bash
claude -c --model sonnet
```

`-c` continues the same session, so the approved plan and all its context carry over. Now
the work is mechanical — write the module, write the test, make it pass — and Sonnet is
faster and cheaper for that.

Say: `Execute phase 0 and phase 1 of the approved plan.`

## Step 3 — the automatic loop

This is what `.claude/hooks/gate.sh` does, and it is the part most people get wrong.

A `Stop` hook runs at the moment Claude Code decides its turn is over. If the hook exits
`2`, Claude is **not allowed to stop** — whatever the hook wrote to stderr is handed back
as its next instruction, and it keeps working. Exit `0` lets it finish.

Our gate runs, in order: ruff, pytest, and from phase 2 onward the retrieval eval gate.
Any failure blocks the stop and hands back the actual failure output. So Claude cannot
end a turn on a red build — it has to fix it first, without you pressing enter.

Two safety catches. Claude Code itself stops honouring a blocking Stop hook after 8
consecutive blocks (`CLAUDE_CODE_STOP_HOOK_BLOCK_CAP`). On top of that our hook keeps its
own counter and surrenders after 5 (`CLAUSE_MAX_LOOPS`), so a problem it cannot solve
comes back to you rather than burning tokens in a circle.

Hooks run on your machine, as you, with your full filesystem and network access. Read
`gate.sh` before you use it — that advice applies to every hook anyone ever hands you,
including this one.

## Step 4 — unattended runs, when you want them

For a long phase you can run headless and walk away:

```bash
claude -p "Execute phase 2 of the approved plan. Stop when make eval writes reports/eval.md and the CI gate passes." \
  --model sonnet \
  --permission-mode acceptEdits \
  --max-turns 40 \
  --output-format stream-json --verbose
```

`--max-turns` caps the loop. There is also `--max-budget-usd` if you want a spending
ceiling. `acceptEdits` auto-approves file edits while still honouring your `deny` list.

## About skipping permissions entirely

`--permission-mode bypassPermissions` (also spelled `--dangerously-skip-permissions`)
turns off every check. Don't. The whole point of `settings.local.json` is that you get
unattended speed *and* a floor — `rm -rf`, `sudo`, force pushes, reading `.env` or your
SSH and cloud credentials stay blocked, and pushes, deploys and `gcloud` still stop to ask
you. That file has `disableBypassPermissionsMode: true` set so the bypass cannot be
turned on by accident.

## What the permissions file actually grants

**Allowed silently:** reading anything in the repo; editing and creating files under
`src/`, `tests/`, `scripts/`, `docs/`, `reports/`, `data/`, plus the top-level config
files; running `uv`, `python`, `pytest`, `ruff`, `mypy`, `make`, ordinary shell reading
tools, local Docker, local `curl`, and git up to and including `commit`; web search; and
fetching from rbi.org.in and a few documentation sites.

**Stops to ask:** `git push`, `gh`, `gcloud`, `docker push`, `pip install`, anything
touching `.env`, and any edit to `.claude/` or `CLAUDE.md` — so the agent can never widen
its own permissions without you seeing it.

**Blocked outright:** `rm -rf`, `sudo`, `chmod 777`, piping curl or wget into a shell,
force push, hard reset, and reading `.env`, `secrets/`, `~/.ssh`, `~/.aws` or your gcloud
config. Reads outside the working directory are blocked too.

Check what is live at any time with `/permissions` inside a session. If a rule does not
behave as written, the settings reference is at
https://code.claude.com/docs/en/settings-reference.md — the format has changed before.
