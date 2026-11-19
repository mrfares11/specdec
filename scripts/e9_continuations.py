"""E9 --- exact greedy continuations and target features, without a GPU.

A quantised copy of the target (llama.cpp, Q5_K_M) drafts each prompt's greedy
continuation quickly; the exact float32 target (scripts/e9_target_pass.py,
layer-streamed) then scores prompt+draft in ONE teacher-forced pass. Wherever
the exact argmax disagrees with the draft, the draft is cut at the first
disagreement, the exact token is inserted, and the quantised model re-drafts
the remainder. The loop ends when every position of the continuation equals
the exact target's argmax given its prefix -- i.e. the continuation is the
float32 target's own greedy output -- and the features recorded in that final
pass are the ones stored. This is speculative decoding with the quantised
model as the drafter and the exact model as the verifier, batched over prompts.

    python scripts/e9_continuations.py --prompts results/e9/prompts.pt \
        --model <safetensors dir> --server <llama-server.exe> --gguf <file> \
        --benchmark mtbench --split test --out <data.pt>
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List

import requests
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.e9_target_pass import StreamingLlama  # noqa: E402

N_CONT = 140  # 128 committed + K(8) + 1 bonus + margin, all verified
PORT = 8089


class Drafter:
    """llama-server wrapper: greedy token continuation from token ids."""

    def __init__(self, exe: Path, gguf: Path, threads: int, slots: int = 4):
        self.exe, self.gguf, self.threads, self.slots = exe, gguf, threads, slots
        self.proc = None

    def start(self):
        if self.proc is not None:
            return
        self.proc = subprocess.Popen(
            [str(self.exe), "-m", str(self.gguf), "-c", str(2048 * self.slots), "-t", str(self.threads),
             "--port", str(PORT), "-np", str(self.slots), "--host", "127.0.0.1"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(600):
            try:
                if requests.get(f"http://127.0.0.1:{PORT}/health", timeout=2).status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(1)
        raise RuntimeError("llama-server did not become healthy")

    def stop(self):
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.proc = None

    def complete(self, ids: List[int], n: int) -> List[int]:
        r = requests.post(f"http://127.0.0.1:{PORT}/completion", json={
            "prompt": ids, "n_predict": n, "temperature": 0.0, "top_k": 1, "ignore_eos": True,
            "return_tokens": True, "cache_prompt": True, "n_probs": 0, "stream": False,
        }, timeout=1800)
        r.raise_for_status()
        toks = r.json().get("tokens")
        if not toks or len(toks) < n:
            raise RuntimeError(f"server returned {len(toks or [])} tokens, wanted {n}")
        return [int(t) for t in toks[:n]]

    def complete_many(self, jobs: List[tuple]) -> Dict[int, List[int]]:
        out = {}
        with ThreadPoolExecutor(max_workers=self.slots) as ex:
            futs = {ex.submit(self.complete, ids, n): key for key, ids, n in jobs}
            for f in futs:
                out[futs[f]] = f.result()
        return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--server", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=14)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    P = torch.load(a.prompts)
    rows = [r for r in P["rows"] if r["benchmark"] == a.benchmark and r["split"] == a.split]
    if a.limit:
        rows = rows[: a.limit]
    eos = int(P["eos_token_id"])
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    state_path = out_path.with_suffix(".state.json")

    # candidate continuations, resumable
    cand: Dict[int, List[int]] = {}
    done: Dict[int, dict] = {}
    if state_path.exists():
        st = json.loads(state_path.read_text())
        cand = {int(k): v for k, v in st.get("cand", {}).items()}
    if out_path.exists():
        for r in torch.load(out_path)["rows"]:
            done[int(r["prompt_id"])] = r
    prompts = {int(r["prompt_id"]): r["input_ids"].tolist() for r in rows}
    pending = [pid for pid in prompts if pid not in done]
    drafter = Drafter(Path(a.server), Path(a.gguf), a.threads)
    sl = StreamingLlama(Path(a.model))
    print(f"[E9] {a.benchmark}/{a.split}: {len(prompts)} prompts, {len(done)} already exact, "
          f"{sum(1 for p in pending if p in cand)} with a candidate", flush=True)
    it = 0
    while pending:
        it += 1
        need = [pid for pid in pending if pid not in cand]
        if need:
            t0 = time.time()
            drafter.start()
            res = drafter.complete_many([(pid, prompts[pid], N_CONT) for pid in need])
            drafter.stop()
            cand.update(res)
            state_path.write_text(json.dumps({"cand": cand}))
            print(f"  iter {it}: drafted {len(need)} candidates in {time.time() - t0:.0f}s", flush=True)
        seqs = [torch.tensor(prompts[pid] + cand[pid], dtype=torch.long) for pid in pending]
        t0 = time.time()
        res = sl.run(seqs, log=lambda m: print("   ", m, flush=True))
        print(f"  iter {it}: exact pass over {len(seqs)} sequences in {time.time() - t0:.0f}s", flush=True)
        still = []
        n_fixed = 0
        for pid, s, r in zip(pending, seqs, res):
            plen = len(prompts[pid])
            am = r["argmax"].tolist()
            # continuation token j sits at position plen + j and must equal argmax[plen + j - 1]
            bad = next((j for j in range(N_CONT) if s[plen + j].item() != am[plen + j - 1]), None)
            if bad is None:
                S = s
                eos_pos = next((j for j in range(N_CONT) if S[plen + j].item() == eos), -1)
                done[pid] = {"benchmark": a.benchmark, "split": a.split, "prompt_id": pid,
                             "prompt_len": plen, "S": S, "feat": r["feat"], "final": r["final"],
                             "eos_offset": eos_pos}
                cand.pop(pid, None)
            else:
                fixed = cand[pid][:bad] + [am[plen + bad - 1]]
                n_fixed += 1
                if len(fixed) >= N_CONT:
                    cand[pid] = fixed[:N_CONT]
                else:
                    cand[pid] = None  # re-draft the remainder below
                    still.append((pid, fixed))
        # re-draft remainders for the prompts cut at a mismatch
        redraft = [(pid, prompts[pid] + fixed, N_CONT - len(fixed)) for pid, fixed in still]
        if redraft:
            drafter.start()
            res2 = drafter.complete_many(redraft)
            drafter.stop()
            for pid, fixed in still:
                cand[pid] = fixed + res2[pid]
        pending = [pid for pid in pending if pid not in done]
        state_path.write_text(json.dumps({"cand": {k: v for k, v in cand.items() if v}}))
        torch.save({"rows": list(done.values()), "eos_token_id": eos}, out_path)
        print(f"  iter {it}: {len(done)}/{len(prompts)} exact; {n_fixed} corrected this pass", flush=True)
    n_eos = sum(1 for r in done.values() if r["eos_offset"] >= 0 and r["eos_offset"] < 128)
    print(f"[E9] done: {len(done)} prompts, {n_eos} hit EOS within 128 tokens; wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
