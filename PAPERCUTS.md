# Papercuts

Small, non-blocking workflow friction recorded in the moment. This is distinct from completed-work logs and tracked bugs.

- [x] 2026-07-22T01:51:20Z **gpt-5** — A combined apply_patch update-and-delete initially failed because I included an extra hunk delimiter before the Delete File directive; clearer validation or an example of mixed operation syntax would avoid the retry.
  Resolved 2026-07-22: Confirmed the current parser accepts mixed operations when the `*** Delete File:` directive follows the completed update hunk directly; `@@` belongs only inside an `*** Update File:` section. A disposable mixed update/delete probe passed, so this was malformed patch input rather than a repository defect and no benchmark source change is warranted.
- [ ] 2026-07-22T05:05:49Z **GPT-5** — While cleaning the exact /tmp/corey-bench-sweep-dry verification directory, the command guard rejected rm -rf even though the target was narrow and explicit. The guard should recommend or permit a recoverable temp-directory cleanup path instead of requiring a retry.
- [ ] 2026-07-22T20:41:34Z **gpt-5** — While validating the existing benchmark harness, 'uv run pytest -q' failed because pytest is not declared in the project's environment even though a pytest-style test suite is checked in. Add a dev/test dependency group or document the supported unittest command.
- [ ] 2026-07-22T21:14:52Z **gpt-5** — While smoke-testing the Quinnferno container, using docker run --rm removed the container immediately after a startup crash and erased the logs needed to diagnose it. Container smoke tests should preserve failed containers until logs are captured.
- [ ] 2026-07-22T21:15:49Z **gpt-5** — While validating the non-root Quinnferno image, BuildKit preserved apply_patch-created source files as mode 0600, so the copied benchmark was unreadable by UID 1001 and Gunicorn crashed. Normalizing application file permissions in the image fixes it; new text files should ideally default to 0644.
