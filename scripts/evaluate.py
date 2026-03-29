"""
Post Evaluation Script

Description:
    Evaluates generated posts using multiple judge models.
    Reads unevaluated rows from the SQLite database and adds judge classifications.

Usage:
    uv run evaluate.py --eval_config <path_to_eval_config.yaml>
    uv run evaluate.py --eval_config <path> --likert_only
    uv run evaluate.py --eval_config <path> --likert_only --judge_filter judgeD judgeE
    uv run evaluate.py --eval_config <path> --filter model=Llama-3.3-70B-Instruct prompt_code=B trial=0
    uv run evaluate.py --eval_config <path> --check_prompt
    uv run evaluate.py --eval_config <path> --dry_run
    uv run evaluate.py --eval_config <path> --opinion_filter A0 A8
    uv run evaluate.py --config <path_to_gen_config.yaml> --likert_only   # legacy
"""

# ============================================================================
# DEPENDENCIES
# ============================================================================
import argparse
import pandas as pd
import yaml
import os
import asyncio
import math
import itertools
from openai import AsyncOpenAI
from tqdm import tqdm
from db import get_connection, init_db, load_df, get_unevaluated_runs, update_evaluations, update_likert_evaluations, ensure_judge_columns

# ============================================================================
# GLOBALS
# ============================================================================
BATCH_SIZE = 25
BINARY_CHOICES = ["0", "1"]
LIKERT_CHOICES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

class RoundRobinClient:
    """Wraps multiple AsyncOpenAI clients and round-robins requests across them."""
    def __init__(self, clients: list[AsyncOpenAI]):
        self._clients = clients
        self._cycle = itertools.cycle(self._clients)

    @property
    def chat(self):
        return next(self._cycle).chat

# ============================================================================
# HELPERS
# ============================================================================
def load_eval_prompt(eval_prompt_path: str) -> str:
    """Load evaluation prompt template."""
    with open(eval_prompt_path, 'r') as file:
        return file.read().strip()

def parse_filter_args(filter_args: list[str] | None) -> dict:
    """Parse --filter key=value pairs into a dict."""
    filters = {}
    if not filter_args:
        return filters
    for item in filter_args:
        key, value = item.split('=', 1)
        if key == 'trial':
            value = int(value)
        filters[key] = value
    return filters

async def evaluate_single_binary(client: AsyncOpenAI, prompt: str, model_name: str, config: dict) -> tuple:
    """
    Send an async request to judge model with guided_choice for binary classification.
    
    Returns:
        tuple: (classification, confidence)
    """
    response = await client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=config['eval_max_completion_tokens'],
        temperature=config['eval_temperature'],
        logprobs=True,
        top_logprobs=2,
        extra_body={"structured_outputs": {"choice": BINARY_CHOICES}},
    )
    
    choice = response.choices[0].message.content.strip()
    classification = choice == "1"
    
    # extract confidence from logprobs
    confidence = 1.0
    if response.choices[0].logprobs and response.choices[0].logprobs.content:
        top_logprob = response.choices[0].logprobs.content[0].logprob
        confidence = math.exp(top_logprob)
    
    return classification, confidence

async def evaluate_single_likert(client: AsyncOpenAI, prompt: str, model_name: str, config: dict, disable_thinking: bool = False) -> tuple:
    """
    Send an async request to judge model with guided_choice for likert scale rating.
    
    Returns:
        tuple: (rating, confidence)
    """
    extra_body = {"structured_outputs": {"choice": LIKERT_CHOICES}}
    if disable_thinking:
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}
        if model_name == "Qwen/Qwen3.5-27B-FP8":  # Qwen3.5 specific override
            extra_body["top_k"] = 20    

    response = await client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=config['eval_max_completion_tokens'],
        temperature=config['eval_temperature'] if model_name != "Qwen/Qwen3.5-27B-FP8" else 0.7,  # Qwen3.5 specific override
        top_p=0.8 if model_name == "Qwen/Qwen3.5-27B-FP8" else None,  # Qwen3.5 specific override
        presence_penalty=1.5 if model_name == "Qwen/Qwen3.5-27B-FP8" else None,  # Qwen3.5 specific override
        logprobs=True,
        top_logprobs=10,
        extra_body=extra_body,
    )
    
    choice = response.choices[0].message.content.strip()
    rating = int(choice)
    
    # extract confidence from logprobs
    confidence = 1.0
    if response.choices[0].logprobs and response.choices[0].logprobs.content:
        top_logprob = response.choices[0].logprobs.content[0].logprob
        confidence = math.exp(top_logprob)
    
    return rating, confidence

async def evaluate_batch(client: AsyncOpenAI, df: pd.DataFrame, eval_prompt: str, judge_config: dict, config: dict, eval_type: str = "binary") -> tuple:
    """
    Evaluate all rows for a single judge using batched async calls.
    
    Args:
        eval_type: "binary" or "likert"
    
    Returns:
        tuple: (results, confidences)
    """
    rows = df[['opinion', 'post']].values.tolist()
    
    all_results = []
    all_confidences = []
    num_batches = (len(rows) + BATCH_SIZE - 1) // BATCH_SIZE
    
    eval_func = evaluate_single_binary if eval_type == "binary" else evaluate_single_likert
    desc_suffix = " (binary)" if eval_type == "binary" else " (likert)"
    disable_thinking = judge_config.get('disable_thinking', False)
    
    for i in tqdm(range(0, len(rows), BATCH_SIZE), total=num_batches, desc=f"  {judge_config['name']}{desc_suffix}"):
        batch = rows[i:i + BATCH_SIZE]
        
        extra_kwargs = {"disable_thinking": disable_thinking} if eval_type == "likert" else {}
        tasks = [
            eval_func(
                client,
                eval_prompt.format(opinion, post),
                judge_config['model_name'],
                config,
                **extra_kwargs
            )
            for opinion, post in batch
        ]
        results = await asyncio.gather(*tasks)
        
        for result, confidence in results:
            all_results.append(result)
            all_confidences.append(confidence)
    
    return all_results, all_confidences

async def main(config: dict, filters: dict = None, check_prompt: bool = False, dry_run: bool = False, opinion_filter: list = None, likert_only: bool = False, judge_filter: list = None):
    """Main async function to run evaluation.
    
    Args:
        opinion_filter: Optional list of opinion_ids to evaluate (e.g., ['A0', 'A8']).
        likert_only: If True, skip binary evaluation and only run Likert scale rating.
        judge_filter: Optional list of judge names to run (subset of config judges).
    """
    eval_binary_prompt = load_eval_prompt(config['eval_binary_prompt_path'])
    eval_likert_prompt = load_eval_prompt(config['eval_likert_prompt_path'])
    
    # open database connection
    conn = get_connection(config['db_path'])
    init_db(conn)
    
    # handle check_prompt mode early (read-only, no schema changes or HTTP calls)
    if check_prompt:
        # validate judge_filter if provided
        if judge_filter:
            matched = [j for j in config['judges'] if j['name'] in judge_filter]
            if not matched:
                print(f"No judges matched --judge_filter {judge_filter}")
                conn.close()
                return
            judge_info = matched[0]
        else:
            judge_info = config['judges'][0]
        
        filter_model = filters.get('model', config.get('eval_model_dir')) if filters else config.get('eval_model_dir')
        pc = filters.get('prompt_code') if filters else None
        tr = filters.get('trial') if filters else None
        df = load_df(conn, model=filter_model, prompt_code=pc, trial=tr)
        df = df[df['post'].notna()].reset_index(drop=True)
        if opinion_filter:
            df = df[df['opinion_id'].isin(opinion_filter)].reset_index(drop=True)
        if df.empty:
            print("No rows with posts found for prompt check.")
            conn.close()
            return
        opinion, post = df.iloc[0]['opinion'], df.iloc[0]['post']
        likert_prompt = eval_likert_prompt.format(opinion, post)
        print(f"\nJudge: {judge_info['name']} ({judge_info['model_name']})")
        print("="*60)
        print("PROMPT CHECK (likert, first row):")
        print("="*60)
        print(likert_prompt)
        print("="*60 + "\n")
        conn.close()
        return
    
    # resolve which judges to run
    active_judges = config['judges']
    if judge_filter:
        active_judges = [j for j in active_judges if j['name'] in judge_filter]
        if not active_judges:
            print(f"No judges matched --judge_filter {judge_filter}")
            return
    
    judge_names = [j['name'] for j in active_judges]
    
    # ensure DB columns exist for all judges in config
    added = ensure_judge_columns(conn, judge_names)
    if added:
        print(f"Added new DB columns: {added}")
    
    # determine runs to process
    filter_model = filters.get('model', config.get('eval_model_dir')) if filters else config.get('eval_model_dir')
    
    if filters and 'prompt_code' in filters and 'trial' in filters:
        # specific run requested
        runs = [(filter_model, filters['prompt_code'], filters['trial'])]
    else:
        # find all unevaluated runs for the active judges
        runs = get_unevaluated_runs(conn, model=filter_model, likert_only=likert_only, judges=judge_names)
    
    if not runs:
        print("No unevaluated runs found.")
        conn.close()
        return
    
    # create clients for each judge (round-robin if multiple ports)
    judge_clients = {}
    for judge in active_judges:
        ports = judge.get('ports', [judge['port']])
        clients = [
            AsyncOpenAI(
                api_key=judge['api_key'],
                base_url=f"{judge['url']}:{p}/v1",
            )
            for p in ports
        ]
        judge_clients[judge['name']] = clients[0] if len(clients) == 1 else RoundRobinClient(clients)

    for model, prompt_code, trial in tqdm(runs, desc="Runs"):
        print(f"\nProcessing: model={model}, prompt={prompt_code}, trial={trial}")
        df = load_df(conn, model=model, prompt_code=prompt_code, trial=trial)
        
        # exclude rows with NULL post (migration placeholders not yet generated)
        df = df[df['post'].notna()].reset_index(drop=True)
        
        # filter to specific opinion_ids if requested
        if opinion_filter:
            df = df[df['opinion_id'].isin(opinion_filter)].reset_index(drop=True)
        
        if df.empty:
            print(f"  No evaluable rows for this run (skipping)")
            continue
        
        # truncate before processing if dry_run
        if dry_run:
            df = df.iloc[:1].copy()
        
        opinion_ids = df['opinion_id'].tolist()
        
        # evaluate with each judge
        for judge in active_judges:
            name = judge['name']
            client = judge_clients[name]
            
            if not likert_only:
                # binary classification
                classifications, confidences = await evaluate_batch(
                    client, df, eval_binary_prompt, judge, config, eval_type="binary"
                )
            
            # likert scale rating
            ratings, rating_confidences = await evaluate_batch(
                client, df, eval_likert_prompt, judge, config, eval_type="likert"
            )
            
            # update database
            if likert_only:
                count = update_likert_evaluations(
                    conn, model, prompt_code, trial, opinion_ids,
                    name, ratings, rating_confidences
                )
            else:
                count = update_evaluations(
                    conn, model, prompt_code, trial, opinion_ids,
                    name, classifications, confidences, ratings, rating_confidences
                )
            print(f"  {name}: updated {count} rows")
        
        print(f"Evaluation complete for {model}/{prompt_code}/trial {trial}")
    
    conn.close()

# ============================================================================
# CONFIG
# ============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default=None,
                    help='(legacy) Path to generation config YAML (contains judge/eval fields)')
parser.add_argument('--eval_config', type=str, default=None,
                    help='Path to standalone evaluation config YAML')
parser.add_argument('--filter', type=str, nargs='+', default=None,
                    help='Filter runs: model=X prompt_code=Y trial=Z')
parser.add_argument('--check_prompt', action='store_true', help='Print first prompt and exit')
parser.add_argument('--dry_run', action='store_true', help='Process only one row per run')
parser.add_argument('--opinion_filter', type=str, nargs='+', default=None,
                    help='Only evaluate these opinion_ids (e.g., A0 A8 B0 B8)')
parser.add_argument('--likert_only', action='store_true',
                    help='Skip binary evaluation and only run Likert scale rating')
parser.add_argument('--judge_filter', type=str, nargs='+', default=None,
                    help='Only run these judges from the config (e.g., judgeD judgeE)')
args = parser.parse_args()

if args.eval_config:
    with open(args.eval_config, 'r') as file:
        config = yaml.safe_load(file)
elif args.config:
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
else:
    parser.error('Must provide --eval_config or --config')

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    filters = parse_filter_args(args.filter)
    asyncio.run(main(config, filters, args.check_prompt, args.dry_run, args.opinion_filter, args.likert_only, args.judge_filter))