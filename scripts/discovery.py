#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from src.discovery import promote, run

parser = argparse.ArgumentParser(description="Quiet Capital broad ETF discovery")
sub = parser.add_subparsers(required=True)
scan = sub.add_parser("daily"); scan.add_argument("--refresh", action="store_true"); scan.add_argument("--top", type=int, default=15)
promotion = sub.add_parser("promote"); promotion.add_argument("ticker"); promotion.add_argument("--reason", required=True); promotion.add_argument("--allocation", type=float, required=True); promotion.add_argument("--horizon", type=int, choices=(63,126), required=True)
args = parser.parse_args()
if args.__dict__.get("ticker"):
    print(json.dumps({"promotion_id": promote(args.ticker, args.reason, args.allocation, args.horizon), "status": "DISCOVERY PAPER TRACK"}, indent=2))
else:
    print(json.dumps(run(args.refresh, top_n=args.top), indent=2))
