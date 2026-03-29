"""
Post Generation Script

Description:
    General project starting point.
    Generates social media posts using a config file that specifies model endpoint, generation parameters, and prompts.
    Saves results to a SQLite database for downstream post evaluation and overton window analysis.

Usage:
    uv run generate.py --config <path_to_config.yaml>
    uv run generate.py --config <path_to_config.yaml> --check_prompt (to print first prompt per topic and exit)
    uv run generate.py --config <path_to_config.yaml> --dry_run (to process only one row per trial)
    uv run generate.py --config <path_to_config.yaml> --prompts baseline --prompt_designation B
    uv run generate.py --config <path_to_config.yaml> --prompts authority foot-in-door --prompt_designation A_FID
    uv run generate.py --config <path_to_config.yaml> --opinion_filter A0 A8 B0 B8 (to generate only for specific opinion_ids)
"""

# ============================================================================
# DEPENDENCIES
# ============================================================================
import argparse
import pandas as pd
import yaml
import os
import asyncio
from openai import AsyncOpenAI
from tqdm import tqdm
from db import get_connection, init_db, insert_generations, upsert_generations, trial_row_count

# ============================================================================
# GLOBALS
# ============================================================================
BATCH_SIZE = 25  # number of concurrent requests per batch

# ============================================================================
# HELPERS
# ============================================================================

def load_fewshot_examples(fewshot_dir: str) -> dict:
    """
    Load all few-shot examples from CSV files into a dictionary.
    
    Args:
        fewshot_dir: Path to directory containing few-shot CSV files.
    
    Returns:
        dict: Mapping from opinion_id prefix (e.g., 'A0') to list of example texts.
    """
    fewshot_map = {}
    
    for filename in os.listdir(fewshot_dir):
        if not filename.endswith('.csv'):
            continue
        
        filepath = os.path.join(fewshot_dir, filename)
        df = pd.read_csv(filepath)
        
        for _, row in df.iterrows():
            idx = row['idx']
            text = row['text']
            # Extract opinion_id prefix (e.g., 'A0' from 'A0_0')
            opinion_prefix = '_'.join(idx.split('_')[:-1])
            
            if opinion_prefix not in fewshot_map:
                fewshot_map[opinion_prefix] = []
            fewshot_map[opinion_prefix].append(text)
    
    return fewshot_map

def initialize_df(schema_path: str, opinions_dir: str) -> pd.DataFrame:
    """
    Create an empty DataFrame with predefined schema, then load opinions from csv files.
    
    Args:
        schema_path: Path to the schema YAML file.
        opinions_dir: Path to directory containing opinion CSV files.
    
    Returns:
        pd.DataFrame: DataFrame with opinions loaded from CSV files.
    """
    with open(schema_path, 'r') as file:
        schema = yaml.safe_load(file)['columns']
    
    # read all CSV files at once (exclude _orig.csv backups and _ex.csv variants)
    opinion_files = [
        os.path.join(opinions_dir, f) 
        for f in os.listdir(opinions_dir) 
        if f.endswith('.csv') and not f.endswith('_orig.csv') and not f.endswith('_ex.csv')
    ]
    
    if not opinion_files:
        return pd.DataFrame({col: pd.Series(dtype=dtype) for col, dtype in schema.items()})
    
    # concatenate all CSVs and rename columns in one step
    df = pd.concat(
        [pd.read_csv(f) for f in opinion_files], 
        ignore_index=True
    ).rename(columns={'idx': 'opinion_id', 'opinion': 'opinion'})
    
    # ensure schema compliance
    for col, dtype in schema.items():
        if col not in df.columns:
            df[col] = pd.Series(dtype=dtype)
        else:
            df[col] = df[col].astype(dtype)

    # sort by opinion_id
    df = df.sort_values(by='opinion_id').reset_index(drop=True)
    
    return df[list(schema.keys())]  # return columns in schema order

def convert_prompt_list(prompt_list: list, prompts_dir: str, exclude_fewshot: bool = False) -> str:
    """
    Convert a list of prompts names into properly formatted prompt string.
    
    Args:
        prompt_list: List of prompt strings.
        prompts_dir: Directory containing prompt files.
        exclude_fewshot: If True, skip the 'few-shot' prompt (handled separately).
    
    Returns:
        str: Concatenated prompt string.
    """
    prompt_contents = []
    for prompt in prompt_list:
        if exclude_fewshot and prompt == 'few-shot':
            continue
        prompt_path = f"{prompts_dir}/{prompt}.txt"
        with open(prompt_path, 'r') as file:
            prompt_content = file.read().strip()
        prompt_contents.append(prompt_content)
    return "\n\n".join(prompt_contents)

def build_prompt_for_opinion(base_prompt: str, opinion: str, opinion_id: str, 
                              fewshot_template: str, fewshot_map: dict) -> str:
    """
    Build the complete prompt for a single opinion, including few-shot examples if applicable.
    
    Args:
        base_prompt: Base prompt template with {opinion} placeholder.
        opinion: The opinion text.
        opinion_id: The opinion identifier (e.g., 'A0', 'H5').
        fewshot_template: The few-shot template string (or None if not using few-shot).
        fewshot_map: Mapping from opinion_id to list of example texts.
    
    Returns:
        str: Complete formatted prompt.
    """
    prompt = base_prompt.format(opinion=opinion)
    
    if fewshot_template and opinion_id in fewshot_map:
        examples = fewshot_map[opinion_id]
        if len(examples) >= 3:
            fewshot_section = fewshot_template.format(
                example_0=examples[0],
                example_1=examples[1],
                example_2=examples[2]
            )
            prompt = prompt + "\n\n" + fewshot_section
    
    return prompt + "\n\nPost: "

async def generate_post(prompt: str, config: dict) -> str:
    """
    Send an async request to the vLLM server for a chat completion.
    
    Args:
        prompt: The formatted prompt string.
        config: Configuration dictionary with generation parameters.
    
    Returns:
        str: Generated post content.
    """
    extra_body = {}
    if config.get('is_thinking_model'):
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}

    response = await ASYNC_CLIENT.chat.completions.create(
        model=config['gen_model_name'],
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=config['gen_max_completion_tokens'],
        temperature=config['gen_temperature'],
        top_p=config['gen_top_p'],
        extra_body=extra_body if extra_body else None,
    )
    return response.choices[0].message.content

async def generate_posts_batch(df: pd.DataFrame, base_prompt: str, config: dict, 
                                fewshot_template: str = None, fewshot_map: dict = None,
                                check_prompt: bool = False, dry_run: bool = False) -> list:
    """
    Generate posts for all opinions in the DataFrame using batched async calls.
    
    Args:
        df: DataFrame containing opinions.
        base_prompt: Base prompt template with {opinion} placeholder.
        config: Configuration dictionary.
        fewshot_template: The few-shot template string (or None if not using few-shot).
        fewshot_map: Mapping from opinion_id to list of example texts.
        check_prompt: If True, print first prompt per topic and exit.
        dry_run: If True, only process the first opinion.
    
    Returns:
        list: Generated posts in order.
    """
    opinions = df['opinion'].tolist()
    opinion_ids = df['opinion_id'].tolist()
    
    if check_prompt:
        # Get first opinion for each topic (identified by first character of opinion_id)
        seen_topics = set()
        for idx, (opinion, oid) in enumerate(zip(opinions, opinion_ids)):
            topic = oid[0]  # First character is the topic code
            if topic in seen_topics:
                continue
            seen_topics.add(topic)
            
            prompt = build_prompt_for_opinion(
                base_prompt, opinion, oid, 
                fewshot_template, fewshot_map or {}
            )
            print("\n" + "="*60)
            print(f"PROMPT CHECK - Topic '{topic}' (opinion_id: {oid}):")
            print("="*60)
            print(prompt)
        print("="*60 + "\n")
        exit(0)
    
    if dry_run:
        opinions = opinions[:1]
        opinion_ids = opinion_ids[:1]
    
    all_posts = []
    # num_batches = (len(opinions) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in range(0, len(opinions), BATCH_SIZE):
        batch_opinions = opinions[i:i + BATCH_SIZE]
        batch_ids = opinion_ids[i:i + BATCH_SIZE]
        
        tasks = [
            generate_post(
                build_prompt_for_opinion(
                    base_prompt, opinion, oid, 
                    fewshot_template, fewshot_map or {}
                ), 
                config
            )
            for opinion, oid in zip(batch_opinions, batch_ids)
        ]
        results = await asyncio.gather(*tasks)
        all_posts.extend(results)
    
    return all_posts

async def main(config: dict, check_prompt: bool = False, dry_run: bool = False, opinion_filter: list = None):
    """
    Main async function to run generation trials.
    
    Args:
        opinion_filter: Optional list of opinion_ids to generate for (e.g., ['A0', 'A8']).
                        When set, uses upsert to backfill existing placeholder rows.
    """
    global ASYNC_CLIENT
    ASYNC_CLIENT = AsyncOpenAI(
        api_key=config['gen_api_key'],
        base_url=f"{config['gen_url']}:{config['gen_port']}/v1",
    )
    
    # Check if few-shot is in the prompts list
    use_fewshot = 'few-shot' in config['prompts']
    
    # Build base prompt, excluding few-shot template (handled separately)
    base_prompt = convert_prompt_list(config['prompts'].copy(), config['prompts_dir'], exclude_fewshot=True)
    
    # Load few-shot template and examples if needed
    fewshot_template = None
    fewshot_map = None
    if use_fewshot:
        fewshot_path = os.path.join(config['prompts_dir'], 'few-shot.txt')
        with open(fewshot_path, 'r') as file:
            fewshot_template = file.read().strip()
        fewshot_map = load_fewshot_examples(config['fewshot_dir'])
    
    # initialize database
    model_suffix = config['gen_model_name'].split('/')[-1]
    conn = get_connection(config['db_path'])
    init_db(conn)
    
    # run trials
    trials = config.get('trials', 1)
    
    for i in tqdm(range(trials), desc=f"Trials [{config['prompt_designation']}]"):

        # initialize DataFrame with opinions
        df = initialize_df(config['schema_path'], config['opinions_dir'])

        # skip trials already fully written to the database
        expected = len(df)
        existing = trial_row_count(conn, model_suffix, config['prompt_designation'], i)
        if existing >= expected and not opinion_filter:
            tqdm.write(f"  Trial {i} already complete ({existing} rows), skipping.")
            continue
        
        # filter to specific opinion_ids if requested
        if opinion_filter:
            df = df[df['opinion_id'].isin(opinion_filter)].reset_index(drop=True)
            if df.empty:
                print(f"  No matching opinion_ids for filter: {opinion_filter}")
                continue
        
        # generate posts for all opinions
        posts = await generate_posts_batch(
            df, base_prompt, config, 
            fewshot_template, fewshot_map,
            check_prompt, dry_run
        )
        
        if dry_run:
            df = df.iloc[:1].copy()
        
        df['post'] = posts
        
        # save results to database (upsert when backfilling filtered opinions)
        rows = list(zip(df['opinion_id'], df['opinion'], df['post']))
        if opinion_filter:
            count = upsert_generations(conn, model_suffix, config['prompt_designation'], i, rows)
            #print(f"Trial {i} complete. Upserted {count} rows into database.")
        else:
            count = insert_generations(conn, model_suffix, config['prompt_designation'], i, rows)
            #print(f"Trial {i} complete. Inserted {count} rows into database.")
    
    conn.close()

# ============================================================================
# CONFIG
# ============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
parser.add_argument('--check_prompt', action='store_true', help='Print first prompt and exit')
parser.add_argument('--dry_run', action='store_true', help='Process only one row per trial')
parser.add_argument('--prompts', type=str, nargs='+', help='Override prompts list from config (space-separated prompt names)')
parser.add_argument('--prompt_designation', type=str, help='Override prompt_designation from config (for filename tagging)')
parser.add_argument('--opinion_filter', type=str, nargs='+', default=None,
                    help='Only generate for these opinion_ids (e.g., A0 A8 B0 B8). Uses upsert for backfill.')
args = parser.parse_args()

with open(args.config, 'r') as file:
    config = yaml.safe_load(file)

# Override config values if command-line arguments provided
if args.prompts is not None:
    config['prompts'] = args.prompts
if args.prompt_designation is not None:
    config['prompt_designation'] = args.prompt_designation

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    asyncio.run(main(config, args.check_prompt, args.dry_run, args.opinion_filter))
