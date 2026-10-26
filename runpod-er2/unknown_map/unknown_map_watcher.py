#!/usr/bin/env python3
"""Watcher for unknown-map exploration: pod sim writes er2_request.json,
this script calls ER2 via Muse Vault helper and writes er2_answer.json back.
Fallback to nearest candidate if ER2 output is invalid or helper fails.
Stops when /workspace/unknown_map/DONE appears on pod, or after 150 minutes.
"""
import ast, json, subprocess, sys, time
from pathlib import Path

SSH = ["ssh", "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=15",
       "-p", "30151", "-i", str(Path.home() / ".ssh" / "id_ed25519"),
       "root@213.173.105.23"]
POD = "/workspace/unknown_map"
HELPER = str(Path.home() / "workspace" / "custom-connectors" / "custom.google-gemini" / "bin" / "gemini.py")
last_id = -1
start = time.time()

def pod_read(path):
    r = subprocess.run(SSH + [f"cat {path}"], capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        return None
    return r.stdout

def pod_write(local_path, remote_path):
    r = subprocess.run(["scp", "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=15",
                        "-P", "30151", "-i", str(Path.home() / ".ssh" / "id_ed25519"),
                        local_path, f"root@213.173.105.23:{remote_path}"],
                       capture_output=True, text=True, timeout=30)
    return r.returncode == 0

def parse_target(raw, candidates):
    # candidates: list of [x, y]
    cand = [tuple(c) for c in candidates]
    try:
        v = ast.literal_eval(raw.strip())
    except Exception:
        return None
    # single tuple (x, y)
    if isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(x, int) for x in v):
        t = (int(v[0]), int(v[1]))
        return t if t in cand else None
    # list of tuples: first one that is a candidate
    if isinstance(v, (list, tuple)):
        for item in v:
            if isinstance(item, (list, tuple)) and len(item) == 2 and all(isinstance(x, int) for x in item):
                t = (int(item[0]), int(item[1]))
                if t in cand:
                    return t
    return None

while time.time() - start < 150 * 60:
    try:
        done = pod_read(f"{POD}/DONE")
        if done is not None:
            print(f"WATCHER done flag on pod: {done.strip()}", flush=True)
            break
        raw_req = pod_read(f"{POD}/er2_request.json")
        if raw_req:
            req = json.loads(raw_req)
            rid = req.get("id")
            if rid is not None and rid != last_id:
                print(f"WATCHER decision {rid} current={req.get('current')} candidates={req.get('candidates')} found={req.get('found')}", flush=True)
                prompt = (
                    "You are exploring an unknown maze. You do NOT know the full map. "
                    "Known free cells are ., known walls are #, unknown cells are ?, "
                    "R is your current cell, digits are gauges already found.\n"
                    f"Partial map rows y=9 to y=0:\n{req.get('partial_map')}\n"
                    f"Current cell: {req.get('current')}. Gauges found so far: {req.get('found')}.\n"
                    f"Candidate frontier cells to explore next (known free cells adjacent to unknown, with BFS distance): {req.get('candidates')}.\n"
                    "Choose ONE candidate that best explores unknown areas and is most likely to lead to new rooms or gauges. "
                    "Prefer a candidate that opens a new direction over one that backtracks, unless backtracking is the only way to an unexplored area. "
                    "Return ONLY a single (x, y) tuple from the candidate list, with no other text, no markdown, and no explanation."
                )
                target = None
                raw_out = ""
                try:
                    r = subprocess.run([sys.executable, HELPER, "er2-waypoints", "--prompt", prompt],
                                       capture_output=True, text=True, timeout=180)
                    if r.returncode == 0:
                        d = json.loads(r.stdout)
                        for s in d.get("steps", []):
                            for c in s.get("content", []):
                                if c.get("type") == "text":
                                    raw_out = c["text"]
                        target = parse_target(raw_out, req.get("candidates", []))
                        print(f"WATCHER er2 raw={raw_out[:200]} parsed={target}", flush=True)
                    else:
                        print(f"WATCHER helper failed rc={r.returncode} stderr={r.stderr[:500]}", flush=True)
                except Exception as e:
                    print(f"WATCHER helper exception {e}", flush=True)
                if target is None:
                    target = tuple(req["candidates"][0])
                    source = "fallback_nearest"
                else:
                    source = "er2"
                ans = {"id": rid, "target": list(target), "source": source, "raw": raw_out}
                tmp = Path("/tmp/unknown_answer.json")
                tmp.write_text(json.dumps(ans))
                ok = pod_write(str(tmp), f"{POD}/er2_answer.json")
                print(f"WATCHER answered id={rid} target={target} source={source} scp_ok={ok}", flush=True)
                last_id = rid
    except Exception as e:
        print(f"WATCHER loop exception {e}", flush=True)
    time.sleep(10)
print("WATCHER exiting", flush=True)
