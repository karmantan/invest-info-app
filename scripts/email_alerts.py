#!/usr/bin/env python3
import argparse
from pathlib import Path
import os
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
if (ROOT/".env").exists():
    for line in (ROOT/".env").read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#") and "=" in line:
            key,value=line.split("=",1); os.environ.setdefault(key.strip(),value.strip().strip('"').strip("'"))

from src.prospective import send_test_email

parser=argparse.ArgumentParser(); parser.add_argument("command",choices=["test"]); args=parser.parse_args()
if args.command=="test":
    try: print(send_test_email())
    except Exception as exc:
        print(f"EMAIL TEST FAILED: {exc}",file=sys.stderr); raise SystemExit(2)
