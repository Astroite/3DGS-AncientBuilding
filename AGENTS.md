# Current workflow entry point

Before operating a capture, Run, reconstruction, or Postshot dataset, read
`docs/CURRENT-WORKFLOW.md`. It is the current native Windows workflow and includes
directory-independent commands, resume rules, the September 2026 run handoff,
and the Postshot Studio CLI license limitation.

Do not infer current commands from the historical WSL/009 demo sections.
Preserve existing uncommitted changes. Run manifests are authoritative; never
mark a failed reconstruction succeeded on disk to make downstream stages run.
An explicitly user-authorized trajectory exception uses the isolated experiment
script described in the current runbook, with an audit record, and is not approval
of reconstruction quality. A previous experiment is not authorization for future
QA bypasses, training, publishing, or sending data to external services.
