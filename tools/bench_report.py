#!/usr/bin/env python3
"""
Bharat AI-SoC PS5 - benchmark collector and report generator.

Runs against the live engine (bin/bharat_qc_native.bin on :5000) and collects:
  * every processed frame (family, defect, severity, pre/infer/post/total ms)
  * every Llama advisory (TTFT, tokens/s, total time, text)
  * system stats once per second (CPU per core, temperature, clock, throttling,
    PMIC power, RSS of the vision and Llama processes)
  * optional idle-power baseline (engine paused)
  * optional image sweep over the sample playlist (prediction vs file-name label)

Writes bench_runs/<timestamp>/{frames,llm,system,sweep}.csv, metadata.json,
summary.json and report.md (ready to paste into the technical report).

Standard library only. Usage (engine already running in another terminal):
  python3 tools/bench_report.py --idle-seconds 30 --seconds 300 --sweep
"""

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

BYPASS_FAMILIES = {"conveyor_belt"}
BYPASS_LABELS = {"no_part"}
CLEAN_WORDS = {"clean", "good", "none", "no_defect"}

# ----------------------------------------------------------------------------- HTTP

class Api:
    def __init__(self, base, timeout=3.0):
        self.base = base.rstrip("/")
        self.timeout = timeout

    def get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=self.timeout) as r:
            return json.loads(r.read().decode("utf-8", "replace") or "{}")

    def state(self):
        return self.get("/api/state")

# ----------------------------------------------------------------------------- stats helpers

def pct(values, p):
    if not values:
        return None
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(p / 100.0 * len(s) + 0.5)) - 1))
    return s[k]

def mean(values):
    return statistics.fmean(values) if values else None

def fmt(v, nd=1):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)

def fps_of(ms):
    return (1000.0 / ms) if ms else None

# ----------------------------------------------------------------------------- system sampler

def read_cpu_times():
    out = {}
    with open("/proc/stat") as f:
        for line in f:
            if line.startswith("cpu"):
                parts = line.split()
                vals = list(map(int, parts[1:]))
                idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
                out[parts[0]] = (sum(vals), idle)
    return out

def read_first(path, cast=str):
    try:
        with open(path) as f:
            return cast(f.read().strip())
    except Exception:
        return None

def find_pids(names):
    found = {}
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        comm = read_first(f"/proc/{pid}/comm")
        if comm in names:
            found.setdefault(comm, []).append(int(pid))
    return found

def rss_mib(pid):
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return None

VCGEN = shutil.which("vcgencmd")

def vcgencmd(*args):
    if not VCGEN:
        return None
    try:
        return subprocess.run([VCGEN, *args], capture_output=True, text=True, timeout=3).stdout
    except Exception:
        return None

def pmic_power_w():
    """Sum of I*V over Pi 5 PMIC rails (vcgencmd pmic_read_adc). Excludes USB/HAT 5 V draw."""
    out = vcgencmd("pmic_read_adc")
    if not out:
        return None
    amps, volts = {}, {}
    for line in out.splitlines():
        m = re.match(r"\s*(\S+)_([AV])\s+\S+=([\d.]+)[AV]", line)
        if m:
            (amps if m.group(2) == "A" else volts)[m.group(1)] = float(m.group(3))
    rails = set(amps) & set(volts)
    return sum(amps[r] * volts[r] for r in rails) if rails else None

def throttled_flags():
    out = vcgencmd("get_throttled")
    if not out or "=" not in out:
        return None
    try:
        return int(out.strip().split("=")[1], 16)
    except ValueError:
        return None

class SystemSampler(threading.Thread):
    PROC_NAMES = {"qc-main-hud", "bharat_qc_nativ", "llama_daemon"}

    def __init__(self, interval=1.0):
        super().__init__(daemon=True)
        self.interval = interval
        self.rows = []
        self.phase = "load"
        self._halt = threading.Event()

    def stop(self):
        self._halt.set()

    def run(self):
        prev = read_cpu_times()
        while not self._halt.wait(self.interval):
            cur = read_cpu_times()
            row = {"t": time.time(), "phase": self.phase}
            for k, (tot, idle) in cur.items():
                if k in prev:
                    dt_tot = tot - prev[k][0]
                    dt_idle = idle - prev[k][1]
                    row[f"{k}_pct"] = round(100.0 * (1 - dt_idle / dt_tot), 1) if dt_tot > 0 else 0.0
            prev = cur
            t = read_first("/sys/class/thermal/thermal_zone0/temp", int)
            row["temp_c"] = t / 1000.0 if t is not None else None
            f = read_first("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", int)
            row["cpu0_mhz"] = f / 1000.0 if f is not None else None
            row["throttled"] = throttled_flags()
            row["pmic_w"] = pmic_power_w()
            pids = find_pids(self.PROC_NAMES)
            vision = [rss_mib(p) for n in ("qc-main-hud", "bharat_qc_nativ") for p in pids.get(n, [])]
            llama = [rss_mib(p) for p in pids.get("llama_daemon", [])]
            row["rss_vision_mib"] = max([v for v in vision if v] or [0]) or None
            row["rss_llama_mib"] = sum(v for v in llama if v) or None
            mem = {}
            try:
                with open("/proc/meminfo") as fm:
                    for line in fm:
                        k, v = line.split(":", 1)
                        mem[k] = int(v.split()[0])
                row["mem_used_mib"] = (mem["MemTotal"] - mem["MemAvailable"]) / 1024.0
            except Exception:
                row["mem_used_mib"] = None
            self.rows.append(row)

# ----------------------------------------------------------------------------- frame / llm capture

def frame_row(s, t):
    tm = s.get("timings_ms") or {}
    an = s.get("anomaly") or {}
    return {
        "t": round(t, 4),
        "frame_id": s.get("frame_id"),
        "mode": s.get("mode"),
        "backend": s.get("vision_backend"),
        "family": s.get("family"),
        "family_conf": s.get("family_conf"),
        "primary_label": s.get("primary_label"),
        "primary_group": s.get("primary_group"),
        "severity": s.get("severity"),
        "verdict": s.get("verdict"),
        "conf": s.get("conf"),
        "n_defects": len(s.get("defects") or []),
        "anomaly_score": an.get("score"),
        "anomaly_status": an.get("status"),
        "pre_ms": tm.get("pre"),
        "infer_ms": tm.get("infer"),
        "post_ms": tm.get("post"),
        "total_ms": tm.get("total"),
    }

def is_bypass(r):
    return (r.get("family") in BYPASS_FAMILIES) or (r.get("primary_label") in BYPASS_LABELS)

def capture(api, seconds, sampler_phase_cb=None, poll=0.03):
    frames, llm = [], []
    seen_frame, last_llm_key = None, None
    errors = 0
    t_end = time.time() + seconds
    while time.time() < t_end:
        try:
            s = api.state()
        except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError):
            errors += 1
            if errors > 50:
                raise SystemExit("Engine API not reachable - is bin/bharat_qc_native.bin running?")
            time.sleep(0.5)
            continue
        now = time.time()
        fid = s.get("frame_id")
        tm = s.get("timings_ms") or {}
        if fid is not None and fid != seen_frame and (tm.get("total") or 0) > 0:
            seen_frame = fid
            frames.append(frame_row(s, now))
        L = s.get("llm") or {}
        key = (L.get("ttft_ms"), L.get("total_ms"), L.get("tokens"))
        if (L.get("total_ms") or 0) > 0 and key != last_llm_key and not s.get("llm_busy"):
            last_llm_key = key
            llm.append({
                "t": round(now, 3),
                "family": s.get("family"),
                "defect": s.get("primary_label"),
                "severity": s.get("severity"),
                "ttft_ms": L.get("ttft_ms"),
                "tokens_per_s": L.get("tokens_per_s"),
                "total_ms": L.get("total_ms"),
                "tokens": L.get("tokens"),
                "source": L.get("source"),
                "cause": L.get("cause"),
                "action": L.get("action"),
            })
        time.sleep(poll)
    if llm:
        llm = llm[1:] if (llm and frames and llm[0]["t"] - frames[0]["t"] < 0.2) else llm  # drop stale advisory present at start
    return frames, llm

# ----------------------------------------------------------------------------- image sweep

def expected_from_name(name):
    base = os.path.splitext(os.path.basename(name))[0].lower()
    base = re.sub(r"^test_\d*_?", "", base)
    neu = {"crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches", "clean_steel"}
    mvtec_nut = ("bent", "color", "flip", "good", "scratch")
    fam_map = {"gc10": "steel_sheet", "weld": "weld", "screw": "machined_part", "casting": "casting",
               "aluminium": "aluminium", "not": "not_metal"}
    if base in neu:
        fam, defect = "steel_sheet", base
    elif base.split("_")[0] in mvtec_nut:
        fam, defect = "machined_part", base.split("_")[0]
    else:
        head = base.split("_")[0]
        fam = fam_map.get(head, "?")
        defect = base[len(head) + 1:] if fam != "not_metal" else "clean"
        if head == "not":
            defect = "clean"
    clean = any(w in defect for w in CLEAN_WORDS)
    return fam, defect, clean

def sweep(api, wait_llm, per_image_timeout=20.0):
    api.get("/api/switch_mode?mode=IMAGE")
    time.sleep(1.0)
    st = api.state()
    playlist = st.get("playlist") or []
    rows = []
    for i, item in enumerate(playlist):
        name = item.get("name") or item.get("path") or f"item{i}"
        before = api.state()
        gen0, fid0 = before.get("sample_gen"), before.get("frame_id")
        api.get(f"/api/select_item?idx={i}")
        t0 = time.time()
        got = None
        while time.time() - t0 < per_image_timeout:
            s = api.state()
            if (s.get("sample_gen") != gen0 and s.get("frame_id") != fid0
                    and (s.get("timings_ms") or {}).get("total", 0) > 0 and s.get("family")):
                got = s
                if not wait_llm:
                    break
                if not s.get("llm_busy") and not s.get("advisory_pending"):
                    break
            time.sleep(0.05)
        exp_fam, exp_defect, exp_clean = expected_from_name(name)
        if got is None:
            rows.append({"idx": i, "image": name, "expected_family": exp_fam, "expected_defect": exp_defect,
                         "status": "TIMEOUT"})
            continue
        r = frame_row(got, time.time())
        label = (r.get("primary_label") or "").lower()
        pred_clean = (label in ("", "none", "no_defect")) and r["n_defects"] == 0 and \
                     (r.get("anomaly_status") != "ANOMALY")
        L = got.get("llm") or {}
        rows.append({
            "idx": i, "image": name,
            "expected_family": exp_fam, "predicted_family": r["family"],
            "family_match": (exp_fam == r["family"]) if exp_fam != "?" else "",
            "expected_defect": exp_defect, "predicted_label": r["primary_label"],
            "predicted_group": r["primary_group"], "severity": r["severity"],
            "conf": r["conf"], "n_defects": r["n_defects"],
            "anomaly_score": r["anomaly_score"], "anomaly_status": r["anomaly_status"],
            "expected_defective": not exp_clean, "predicted_defective": not pred_clean,
            "defect_presence_match": (not exp_clean) == (not pred_clean),
            "total_ms": r["total_ms"], "pre_ms": r["pre_ms"], "infer_ms": r["infer_ms"], "post_ms": r["post_ms"],
            "llm_total_ms": L.get("total_ms") if wait_llm else "",
            "advisory": (f"{L.get('cause','')} | {L.get('action','')}" if wait_llm else ""),
            "status": "OK",
        })
    return rows

# ----------------------------------------------------------------------------- metadata

def run_cmd(cmd, timeout=10):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, shell=isinstance(cmd, str)).stdout.strip()
    except Exception as e:
        return f"unavailable ({e.__class__.__name__})"

def sha256(path):
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None

def collect_metadata(repo):
    os_name = "?"
    try:
        with open("/etc/os-release") as f:
            for line in f:
                if line.startswith("PRETTY_NAME="):
                    os_name = line.split("=", 1)[1].strip().strip('"')
    except Exception:
        pass
    return {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "board": (read_first("/proc/device-tree/model") or "?").replace("\x00", ""),
        "os": os_name,
        "kernel": platform.release(),
        "hailortcli_identify": run_cmd(["hailortcli", "fw-control", "identify"]),
        "git_commit": run_cmd(["git", "-C", repo, "rev-parse", "--short", "HEAD"]),
        "git_dirty_files": run_cmd(["git", "-C", repo, "status", "--porcelain"]).count("\n") + 1
                           if run_cmd(["git", "-C", repo, "status", "--porcelain"]) else 0,
        "hef_sha256": sha256(os.path.join(repo, "models/unified_yolo11_640.hef")),
        "config_sha256": sha256(os.path.join(repo, "config/edge_qc_config.json")),
        "vcgencmd": bool(VCGEN),
    }

# ----------------------------------------------------------------------------- output

def write_csv(path, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

def summarize(frames, llm, sysrows, sweep_rows, wall_s):
    metal = [r for r in frames if not is_bypass(r) and r.get("total_ms")]
    bypass = [r for r in frames if is_bypass(r) and r.get("total_ms")]
    fams = sorted({r["family"] for r in metal if r.get("family")})

    def lat_block(rows):
        tot = [r["total_ms"] for r in rows]
        return {
            "n": len(rows),
            "total_mean": mean(tot), "total_p50": pct(tot, 50), "total_p95": pct(tot, 95), "total_max": max(tot) if tot else None,
            "fps_mean": fps_of(mean(tot)), "fps_p95": fps_of(pct(tot, 95)),
            "pre_mean": mean([r["pre_ms"] for r in rows if r.get("pre_ms") is not None]),
            "infer_mean": mean([r["infer_ms"] for r in rows if r.get("infer_ms") is not None]),
            "post_mean": mean([r["post_ms"] for r in rows if r.get("post_ms") is not None]),
        }

    fid = [r["frame_id"] for r in frames if isinstance(r.get("frame_id"), int)]
    processed = (max(fid) - min(fid) + 1) if len(fid) > 1 else None
    defect_counts = {}
    for r in metal:
        key = (r.get("family"), r.get("primary_label") or "NONE")
        defect_counts[key] = defect_counts.get(key, 0) + 1

    def sys_block(phase):
        rows = [r for r in sysrows if r["phase"] == phase]
        if not rows:
            return None
        cores = sorted((k for k in rows[0] if re.match(r"cpu\d+_pct", k)), key=lambda k: int(re.findall(r"\d+", k)[0]))
        thr = [r["throttled"] for r in rows if r.get("throttled") is not None]
        pw = [r["pmic_w"] for r in rows if r.get("pmic_w") is not None]
        return {
            "samples": len(rows),
            "cpu_total_mean": mean([r.get("cpu_pct", 0) for r in rows]),
            "cpu_core_mean": {c.replace("_pct", ""): mean([r.get(c, 0) for r in rows]) for c in cores},
            "temp_mean": mean([r["temp_c"] for r in rows if r.get("temp_c") is not None]),
            "temp_max": max([r["temp_c"] for r in rows if r.get("temp_c") is not None] or [0]) or None,
            "cpu0_mhz_min": min([r["cpu0_mhz"] for r in rows if r.get("cpu0_mhz")] or [0]) or None,
            "throttled_any": (hex(max(thr)) if thr else None),
            "pmic_w_mean": mean(pw), "pmic_w_max": max(pw) if pw else None,
            "rss_vision_max": max([r["rss_vision_mib"] for r in rows if r.get("rss_vision_mib")] or [0]) or None,
            "rss_llama_max": max([r["rss_llama_mib"] for r in rows if r.get("rss_llama_mib")] or [0]) or None,
            "mem_used_max": max([r["mem_used_mib"] for r in rows if r.get("mem_used_mib")] or [0]) or None,
        }

    llm_ok = [a for a in llm if a.get("total_ms")]
    sweep_ok = [r for r in sweep_rows if r.get("status") == "OK"]
    return {
        "wall_seconds": wall_s,
        "frames_captured": len(frames),
        "frames_processed_by_engine": processed,
        "engine_throughput_fps": (processed / wall_s) if processed and wall_s else None,
        "backends": sorted({r.get("backend") for r in frames if r.get("backend")}),
        "metal_all": lat_block(metal),
        "per_family": {f: lat_block([r for r in metal if r.get("family") == f]) for f in fams},
        "bypass": lat_block(bypass),
        "defect_counts": {f"{k[0]} / {k[1]}": v for k, v in sorted(defect_counts.items())},
        "llm": {
            "n": len(llm_ok),
            "ttft_mean": mean([a["ttft_ms"] for a in llm_ok]), "ttft_p95": pct([a["ttft_ms"] for a in llm_ok], 95),
            "tps_mean": mean([a["tokens_per_s"] for a in llm_ok]),
            "total_mean": mean([a["total_ms"] for a in llm_ok]), "total_p95": pct([a["total_ms"] for a in llm_ok], 95),
            "tokens_mean": mean([a["tokens"] for a in llm_ok if a.get("tokens") is not None]),
        },
        "system_idle": sys_block("idle"),
        "system_load": sys_block("load"),
        "sweep": {
            "images": len(sweep_rows), "ok": len(sweep_ok),
            "family_correct": sum(1 for r in sweep_ok if r.get("family_match") is True),
            "family_scored": sum(1 for r in sweep_ok if r.get("family_match") in (True, False)),
            "defect_presence_correct": sum(1 for r in sweep_ok if r.get("defect_presence_match")),
        } if sweep_rows else None,
    }

def report_md(meta, S, sweep_rows, args):
    L = []
    a = L.append
    a("# Edge Benchmark Report - Bharat AI-SoC PS5\n")
    a(f"Generated {meta['timestamp']} on **{meta['board']}** ({meta['os']}, kernel {meta['kernel']}).  ")
    a(f"Git commit `{meta['git_commit']}` ({meta['git_dirty_files']} locally modified files), "
      f"HEF sha256 `{(meta['hef_sha256'] or '-')[:12]}`, config sha256 `{(meta['config_sha256'] or '-')[:12]}`.\n")
    a(f"Load run: **{fmt(S['wall_seconds'],0)} s** in DEMO mode; vision backend(s): **{', '.join(S['backends']) or '-'}**.\n")

    a("## 1. Throughput and latency (metal frames)\n")
    a("Empty-belt frames are excluded here and reported separately in 1.3. FPS = 1000 / latency for one frame.\n")
    a("| Family | Frames | Mean ms | p50 ms | p95 ms | Max ms | FPS (mean) | FPS (p95) |")
    a("|---|---|---|---|---|---|---|---|")
    for fam, b in list(S["per_family"].items()) + [("**All metal**", S["metal_all"])]:
        a(f"| {fam} | {b['n']} | {fmt(b['total_mean'])} | {fmt(b['total_p50'])} | {fmt(b['total_p95'])} | "
          f"{fmt(b['total_max'])} | {fmt(b['fps_mean'])} | {fmt(b['fps_p95'])} |")
    a("")
    a("### 1.2 Stage breakdown (mean ms)\n")
    a("`infer` = all YOLO passes on the Hailo-8L (tiles + full frame) plus PatchCore scoring on the CPU.\n")
    a("| Family | Pre | Infer | Post | Total |")
    a("|---|---|---|---|---|")
    for fam, b in list(S["per_family"].items()) + [("**All metal**", S["metal_all"])]:
        a(f"| {fam} | {fmt(b['pre_mean'])} | {fmt(b['infer_mean'])} | {fmt(b['post_mean'])} | {fmt(b['total_mean'])} |")
    a("")
    bp = S["bypass"]
    a("### 1.3 Empty-belt bypass\n")
    a(f"{bp['n']} frames, mean {fmt(bp['total_mean'])} ms (p95 {fmt(bp['total_p95'])} ms, ~{fmt(bp['fps_mean'],0)} FPS). "
      "No detector or LLM runs on these frames.\n")
    a(f"Engine processed **{S['frames_processed_by_engine'] or '-'}** frames in {fmt(S['wall_seconds'],0)} s "
      f"(**{fmt(S['engine_throughput_fps'])} frames/s** wall-clock, including empty belt and LLM pauses); "
      f"{S['frames_captured']} of them were sampled by this script.\n")

    l = S["llm"]
    a("## 2. On-device Llama 3.2 1B INT4 (ExecuTorch, cores 2-3)\n")
    a("| Advisories | TTFT mean | TTFT p95 | Tokens/s | Total mean | Total p95 | Tokens/advisory |")
    a("|---|---|---|---|---|---|---|")
    a(f"| {l['n']} | {fmt(l['ttft_mean'],0)} ms | {fmt(l['ttft_p95'],0)} ms | {fmt(l['tps_mean'])} | "
      f"{fmt(l['total_mean'],0)} ms | {fmt(l['total_p95'],0)} ms | {fmt(l['tokens_mean'])} |")
    a("\nMeasured while vision runs concurrently on the same SoC.\n")

    a("## 3. System resources\n")
    a("| Metric | Idle (engine paused) | Under load (DEMO) |")
    a("|---|---|---|")
    si, sl = S["system_idle"] or {}, S["system_load"] or {}
    def row(name, key, nd=1, unit=""):
        a(f"| {name} | {fmt(si.get(key), nd)}{unit if si.get(key) is not None else ''} | "
          f"{fmt(sl.get(key), nd)}{unit if sl.get(key) is not None else ''} |")
    row("CPU total (mean)", "cpu_total_mean", 1, " %")
    row("SoC temperature (mean)", "temp_mean", 1, " °C")
    row("SoC temperature (max)", "temp_max", 1, " °C")
    row("CPU0 clock (min)", "cpu0_mhz_min", 0, " MHz")
    row("PMIC power (mean)", "pmic_w_mean", 2, " W")
    row("PMIC power (max)", "pmic_w_max", 2, " W")
    row("Vision process RSS (max)", "rss_vision_max", 0, " MiB")
    row("Llama process RSS (max)", "rss_llama_max", 0, " MiB")
    row("System memory used (max)", "mem_used_max", 0, " MiB")
    a(f"| Throttle flags seen | {si.get('throttled_any') or '-'} | {sl.get('throttled_any') or '-'} |")
    if sl.get("cpu_core_mean"):
        a("\nPer-core CPU under load (mean %): " +
          ", ".join(f"{k} {fmt(v)}" for k, v in sl["cpu_core_mean"].items()) + ".\n")
    a("PMIC power is the sum of the Pi 5 PMIC rails (`vcgencmd pmic_read_adc`); it does not include the 5 V "
      "drawn by the AI HAT/USB, so quote a USB-C inline meter reading for total board power.\n")

    if sweep_rows:
        sw = S["sweep"]
        a("## 4. Sample-image sweep (IMAGE mode)\n")
        a(f"{sw['ok']}/{sw['images']} images processed. Material family correct: "
          f"**{sw['family_correct']}/{sw['family_scored']}**. Defect presence (defective vs clean) correct: "
          f"**{sw['defect_presence_correct']}/{sw['ok']}**. Expected labels come from the file names; this is a "
          "smoke test on a small set, not a precision/recall evaluation.\n")
        a("| # | Image | Expected family | Predicted family | Expected defect | Predicted label (group) | Severity | Conf | Total ms |")
        a("|---|---|---|---|---|---|---|---|---|")
        for r in sweep_rows:
            if r.get("status") != "OK":
                a(f"| {r['idx']} | {r['image']} | {r['expected_family']} | TIMEOUT | {r['expected_defect']} | - | - | - | - |")
                continue
            fm = "" if r["family_match"] in (True, "") else " ✗"
            dm = "" if r["defect_presence_match"] else " ✗"
            a(f"| {r['idx']} | {r['image']} | {r['expected_family']} | {r['predicted_family']}{fm} | {r['expected_defect']} | "
              f"{r['predicted_label'] or 'NONE'} ({r['predicted_group'] or '-'}){dm} | {r['severity']} | {fmt(r['conf'],2)} | {fmt(r['total_ms'])} |")
        a("")

    a("## 5. Method\n")
    a(f"- Engine `bin/bharat_qc_native.bin` running on the device; this script polled `/api/state` every "
      f"{int(args.poll*1000)} ms and recorded each new `frame_id` (frames between polls are not sampled; "
      "latencies are per-frame values reported by the engine's own steady_clock timers).")
    a(f"- Idle baseline: {args.idle_seconds} s with the engine paused. Load: {args.seconds} s of DEMO conveyor playback.")
    a("- Raw data: `frames.csv`, `llm.csv`, `system.csv`, `sweep.csv`, `bench_api_end.json`, `metadata.json`.")
    a(f"- Hailo device: `{(meta['hailortcli_identify'] or '').splitlines()[0] if meta['hailortcli_identify'] else '-'}`")
    return "\n".join(L) + "\n"

# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:5000")
    ap.add_argument("--seconds", type=int, default=300, help="DEMO load capture length")
    ap.add_argument("--idle-seconds", type=int, default=30, help="idle baseline with engine paused (0 = skip)")
    ap.add_argument("--sweep", action="store_true", help="also run the IMAGE-mode sweep over the sample playlist")
    ap.add_argument("--sweep-wait-llm", action="store_true", help="wait for each advisory during the sweep (slower)")
    ap.add_argument("--poll", type=float, default=0.03, help="state poll interval in seconds")
    ap.add_argument("--out", default="bench_runs")
    ap.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    args = ap.parse_args()

    api = Api(args.url)
    try:
        st = api.state()
    except Exception as e:
        sys.exit(f"Cannot reach {args.url}/api/state ({e}). Start ./setup_and_run_rpi5.sh first.")
    print(f"[bench] engine reachable: backend={st.get('vision_backend')} mode={st.get('mode')}")

    out = os.path.join(args.out, dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out, exist_ok=True)
    meta = collect_metadata(args.repo)
    sampler = SystemSampler()
    sampler.start()

    if st.get("mode") != "DEMO":
        api.get("/api/switch_mode?mode=DEMO")
        time.sleep(1.0)

    if args.idle_seconds > 0:
        if not api.state().get("is_paused"):
            api.get("/api/toggle_pause")
        sampler.phase = "idle"
        print(f"[bench] idle baseline {args.idle_seconds}s (engine paused)...")
        time.sleep(args.idle_seconds)
        if api.state().get("is_paused"):
            api.get("/api/toggle_pause")
        time.sleep(1.0)

    sampler.phase = "load"
    print(f"[bench] DEMO load capture {args.seconds}s ...")
    t0 = time.time()
    frames, llm = capture(api, args.seconds, poll=args.poll)
    wall = time.time() - t0
    try:
        bench_end = api.get("/api/bench")
    except Exception:
        bench_end = {}

    sweep_rows = []
    if args.sweep:
        sampler.phase = "sweep"
        print("[bench] IMAGE sweep over the sample playlist ...")
        sweep_rows = sweep(api, args.sweep_wait_llm)
        api.get("/api/switch_mode?mode=DEMO")

    sampler.stop()
    sampler.join(timeout=3)

    S = summarize(frames, llm, sampler.rows, sweep_rows, wall)
    write_csv(os.path.join(out, "frames.csv"), frames)
    write_csv(os.path.join(out, "llm.csv"), llm)
    write_csv(os.path.join(out, "system.csv"), sampler.rows)
    write_csv(os.path.join(out, "sweep.csv"), sweep_rows)
    with open(os.path.join(out, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(out, "bench_api_end.json"), "w") as f:
        json.dump(bench_end, f, indent=2)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(S, f, indent=2, default=str)
    with open(os.path.join(out, "report.md"), "w") as f:
        f.write(report_md(meta, S, sweep_rows, args))

    m = S["metal_all"]
    print(f"[bench] metal frames: n={m['n']} mean={fmt(m['total_mean'])} ms p95={fmt(m['total_p95'])} ms "
          f"-> {fmt(m['fps_mean'])} FPS (p95 {fmt(m['fps_p95'])})")
    print(f"[bench] llm advisories: {S['llm']['n']}  TTFT {fmt(S['llm']['ttft_mean'],0)} ms  {fmt(S['llm']['tps_mean'])} tok/s")
    print(f"[bench] report: {os.path.join(out, 'report.md')}")

if __name__ == "__main__":
    main()
