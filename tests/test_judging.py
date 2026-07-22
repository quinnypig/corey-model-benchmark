import unittest
import tempfile

from corey_bench.judging import JudgeOutputError, judge_messages, parse_judge_output
from corey_bench.openrouter import Completion
from corey_bench.protocol import load_protocol
from corey_bench.runner import ReviewJob, RunConfig, RunQueue, RunStore, read_jsonl


class FakeJudgeClient:
    def complete_messages(self, **kwargs):
        return Completion(
            text='{"score":0.75,"verdict":"solid","rationale":"used the rubric","rubric_scores":[]}',
            response_id="judge-1", provider="fixture", usage={"cost": 0.012}, raw_model="judge/model",
            reasoning=None, finish_reason="stop", native_finish_reason="stop", annotations=[],
        )


class JudgingTests(unittest.TestCase):
    def test_parses_fenced_rubric_receipt(self):
        parsed = parse_judge_output(
            '```json\n{"score":0.8,"verdict":"good","rationale":"specific",'
            '"rubric_scores":[{"name":"design","score":0.8,"rationale":"clean"}]}\n```'
        )
        self.assertEqual(parsed["score"], 0.8)
        self.assertEqual(parsed["rubric_scores"][0]["name"], "design")

    def test_rejects_judge_output_without_score(self):
        with self.assertRaises(JudgeOutputError):
            parse_judge_output('{"verdict":"shrug"}')

    def test_prompt_omits_candidate_model_identity(self):
        messages = judge_messages(
            eval_manifest={"id": "1.2", "title": "Platypus", "prompts": ["draw it"], "rubric": []},
            result={"model": "secret/model", "responses": ["<svg/>"]},
        )
        self.assertNotIn("secret/model", str(messages))
        self.assertIn("overall_quality", messages[1]["content"])

    def test_review_job_writes_auditable_model_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            store = RunStore(directory)
            run_id, jobs = store.create(
                RunConfig(models=["candidate/model"], eval_ids=["1.1"], conditions=["weights-only"], repetitions=1),
                load_protocol(),
            )
            store.append_result(
                run_id,
                {
                    "attempt_id": jobs[0].attempt_id, "status": "ok", "model": "candidate/model",
                    "eval_id": "1.1", "condition": "weights-only", "response": "Taft is absorbent.",
                    "responses": ["Taft is absorbent."], "cost_usd": 0.01,
                    "grade": {"score": 0.5, "human_required": True},
                },
            )
            queue = RunQueue("test", store=store, judge_model="judge/model", judge_workers=0)
            queue.client = FakeJudgeClient()
            queue._execute_review(ReviewJob(run_id, jobs[0].attempt_id))
            receipts = read_jsonl(store.run_dir(run_id) / "reviews.jsonl")
            self.assertEqual(receipts[0]["reviewer_type"], "model")
            self.assertEqual(receipts[0]["score"], 0.75)
            self.assertEqual(receipts[0]["cost_usd"], 0.012)
