# Papercuts

Small, non-blocking workflow friction recorded in the moment. This is distinct from completed-work logs and tracked bugs.

- [x] 2026-07-22T01:51:20Z **gpt-5** — A combined apply_patch update-and-delete initially failed because I included an extra hunk delimiter before the Delete File directive; clearer validation or an example of mixed operation syntax would avoid the retry.
  Resolved 2026-07-22: Confirmed the current parser accepts mixed operations when the `*** Delete File:` directive follows the completed update hunk directly; `@@` belongs only inside an `*** Update File:` section. A disposable mixed update/delete probe passed, so this was malformed patch input rather than a repository defect and no benchmark source change is warranted.
- [ ] 2026-07-22T05:05:49Z **GPT-5** — While cleaning the exact /tmp/corey-bench-sweep-dry verification directory, the command guard rejected rm -rf even though the target was narrow and explicit. The guard should recommend or permit a recoverable temp-directory cleanup path instead of requiring a retry.
