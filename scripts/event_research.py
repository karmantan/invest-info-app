import argparse
from src.reporting.event_research import ingest, run

parser=argparse.ArgumentParser(); parser.add_argument("command",choices=["ingest","run"]); parser.add_argument("--refresh",action="store_true"); args=parser.parse_args()
ingest(args.refresh) if args.command=="ingest" else run()
