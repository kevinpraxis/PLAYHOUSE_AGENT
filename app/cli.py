# app/cli.py
# -*- coding: utf-8 -*-
import os
import typer
from typing import Optional
from dotenv import load_dotenv

def main(
    query: Optional[str] = typer.Option(None, help="Override search query; default uses built-in QUERIES"),
    target: int = typer.Option(int(os.getenv("TARGET", "6")), min=1, help="Max candidates to process"),
    dry_run: bool = typer.Option(True, "--dry-run/--live", help="Dry run = no posting"),
    interactive: bool = typer.Option(False, "--interactive/--no-interactive", help="Human review before posting"),
):
    # 
    load_dotenv(override=False)
    # 
    from app.main import run_once
    # 
    run_once(query=query, target=target, dry_run=dry_run, interactive=interactive)

if __name__ == "__main__":
    typer.run(main)
