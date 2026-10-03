#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, fcntl, json, signal, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.spreads import collect,load_spread_config
from src.storage import Database

def _timeout(signum,frame): raise TimeoutError("spread collection exceeded its bounded timeout")

def cmd_collect(args):
    cfg=load_spread_config(); lock_path=ROOT/"state/spreads.lock"; lock_path.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open("w") as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: print(json.dumps({"status":"LOCKED","reason":"another spread collector is running"})); return 0
        signal.signal(signal.SIGALRM,_timeout); signal.alarm(int(cfg["timeout_seconds"]))
        try: result=collect(force=args.force)
        except TimeoutError as exc:
            result={"status":"FAILED","reason":str(exc),"requests":"unknown"}
        finally: signal.alarm(0)
    print(json.dumps(result,indent=2)); return 0 if result["status"] in {"COMPLETE","SKIPPED","COOLDOWN"} else 1

def cmd_status(_args):
    db=Database(); db.initialize(); cfg=load_spread_config()
    with db.connect() as c:
        latest=c.execute("SELECT * FROM spread_attempts ORDER BY id DESC LIMIT 1").fetchone()
        cooldown=c.execute("SELECT value FROM spread_collector_state WHERE key='cooldown_until_utc'").fetchone()
    print(json.dumps({"configured_mappings":cfg["mappings"],"latest_attempt":dict(latest) if latest else None,
                      "cooldown_until_utc":cooldown[0] if cooldown else None},indent=2,default=str)); return 0

def cmd_export(args):
    db=Database(); db.initialize(); output=Path(args.output)
    with db.connect() as c:
        rows=c.execute("SELECT * FROM spread_observations WHERE available_at_utc<=? ORDER BY available_at_utc,id",
                       (args.as_of or datetime.now(timezone.utc).isoformat(),)).fetchall()
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=rows[0].keys() if rows else ["id"]); writer.writeheader(); writer.writerows(dict(r) for r in rows)
    print(json.dumps({"output":str(output),"observations":len(rows),"as_of_utc":args.as_of})); return 0

def main():
    parser=argparse.ArgumentParser(description="Collect and inspect indicative European-listing spreads")
    commands=parser.add_subparsers(dest="command",required=True)
    collect_p=commands.add_parser("collect"); collect_p.add_argument("--force",action="store_true",help="probe outside market time (still validates/stores result)"); collect_p.set_defaults(func=cmd_collect)
    commands.add_parser("status").set_defaults(func=cmd_status)
    export_p=commands.add_parser("export"); export_p.add_argument("--output",default=str(ROOT/"reports/spread_observations.csv")); export_p.add_argument("--as-of",help="UTC availability cutoff (ISO-8601)"); export_p.set_defaults(func=cmd_export)
    args=parser.parse_args(); raise SystemExit(args.func(args))

if __name__=="__main__": main()
