#!/usr/bin/env python3
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.model_governance import create_challenger,initialize_versions,require_structural_review,status

parser=argparse.ArgumentParser(); sub=parser.add_subparsers(dest="command",required=True)
sub.add_parser("initialize"); sub.add_parser("status")
c=sub.add_parser("create-challenger"); c.add_argument("--reason",choices=["SCHEDULED_RETRAIN","ANNUAL_REVIEW","STRUCTURAL_BREAK_REVIEW"],default="SCHEDULED_RETRAIN")
r=sub.add_parser("structural-review"); r.add_argument("--create-challenger",action="store_true")
args=parser.parse_args()
if args.command=="initialize": initialize_versions(); print(json.dumps(status(),indent=2))
elif args.command=="status": print(json.dumps(status(),indent=2))
elif args.command=="create-challenger": print(create_challenger(args.reason))
else: print(require_structural_review(args.create_challenger))
