#!/usr/bin/env python3
import argparse, json
from src.prospective import run_daily

parser=argparse.ArgumentParser(); parser.add_argument("command",choices=["daily"]); parser.add_argument("--ingestion-failed")
args=parser.parse_args()
if args.command=="daily": print(json.dumps(run_daily(args.ingestion_failed),indent=2))
