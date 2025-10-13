"""
LLM-as-judge evaluation of Frotzmark transcripts.

Sends a completed gameplay transcript to a judge model for qualitative assessment.
"""

import sys
import re
import shutil
from pathlib import Path
from typing import Optional

import click
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from .config import API_KEY, BASE_URL, PROVIDER_TYPE


def wrap_and_echo(text: str) -> None:
    """
    Wrap text to terminal width and echo it using Click.

    Preserves all newlines (both single and double) from the original text,
    only wrapping long lines to fit terminal width.

    Args:
        text: Text to wrap and display
    """
    # Get terminal width
    try:
        width = shutil.get_terminal_size().columns
    except:
        width = 80  # fallback

    # Split into lines, wrap each line individually, preserving all newlines
    lines = text.split('\n')
    wrapped_lines = []
    for line in lines:
        if line:  # Non-empty line - wrap it
            wrapped = click.wrap_text(line, width=width, preserve_paragraphs=False)
            wrapped_lines.append(wrapped)
        else:  # Empty line - preserve it
            wrapped_lines.append('')

    wrapped = '\n'.join(wrapped_lines)
    click.echo(wrapped)


def parse_frontmatter(transcript_text: str) -> dict:
    """
    Extract YAML frontmatter from transcript.

    Returns:
        Dictionary of frontmatter values
    """
    frontmatter_match = re.match(r'^---\n(.*?)\n---', transcript_text, re.DOTALL)
    if not frontmatter_match:
        return {}

    frontmatter_text = frontmatter_match.group(1)
    frontmatter = {}

    # Parse simple key: value pairs (handles quoted strings and bare values)
    for line in frontmatter_text.split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()

            # Remove quotes if present
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            elif value.startswith("'") and value.endswith("'"):
                value = value[1:-1]

            frontmatter[key] = value

    return frontmatter


def create_evaluation_prompt(transcript: str, frontmatter: dict) -> str:
    """
    Create the evaluation prompt for the judge model.

    Args:
        transcript: Full transcript text
        frontmatter: Parsed frontmatter metadata

    Returns:
        Evaluation prompt string
    """
    model_name = frontmatter.get('model', 'unknown')
    objective = frontmatter.get('objective')
    max_turns = frontmatter.get('max_turns', 'unknown')

    prompt = f"""You are evaluating a language model's performance playing an interactive fiction game.

# Transcript Metadata
- Model: {model_name}
- Max turns: {max_turns}"""

    if objective:
        prompt += f"\n- Objective: {objective}"
    else:
        prompt += "\n- Objective: Maximize score"

    prompt += f"""

# Your Task
Analyze this gameplay transcript and provide a qualitative assessment of the model's performance. Consider:

1. **Goal Achievement**: Did the model achieve its objective? How efficiently?
2. **Strategic Thinking**: Did the model demonstrate systematic exploration and planning?
3. **Error Recovery**: How did the model handle mistakes, dead ends, or unexpected responses?
4. **Planning Quality**: Were the model's <planning> blocks coherent and useful?
5. **Notable Behaviors**: Any interesting patterns, creative solutions, or failure modes?

Be specific and cite examples from the transcript. Your assessment should help understand *why* the model performed as it did, not just *what* it did.

# Transcript
{transcript}

# Your Assessment
"""

    return prompt


@click.command()
@click.argument('transcript_path', type=click.Path(exists=True, path_type=Path))
@click.option('--judge-model', default='openai/gpt-5-mini', help='Model to use for evaluation (default: openai/gpt-5-mini)')
@click.option('--output', '-o', type=click.Path(path_type=Path), help='Write evaluation to file instead of stdout')
def main(
    transcript_path: Path,
    judge_model: str,
    output: Optional[Path],
) -> None:
    """
    Evaluate a Frotzmark gameplay transcript using an LLM judge.

    TRANSCRIPT_PATH: Path to the transcript markdown file to evaluate

    The judge model will provide qualitative assessment of the model's
    performance, strategic thinking, and notable behaviors.
    """

    click.echo(f"📊 Evaluating transcript: {transcript_path.name}")
    click.echo(f"🤖 Judge model: {judge_model}\n")

    # Read transcript
    try:
        transcript_text = transcript_path.read_text()
    except Exception as e:
        click.echo(f"Error reading transcript: {e}", err=True)
        sys.exit(1)

    # Parse frontmatter
    frontmatter = parse_frontmatter(transcript_text)
    tested_model = frontmatter.get('model', 'unknown')
    click.echo(f"📝 Tested model: {tested_model}")

    # Create judge agent
    if PROVIDER_TYPE == "openrouter":
        provider = OpenAIProvider(api_key=API_KEY, base_url=BASE_URL)
        model = OpenAIChatModel(judge_model, provider=provider)
    else:
        provider = OpenAIProvider(api_key=API_KEY, base_url=BASE_URL)
        model = OpenAIChatModel(judge_model, provider=provider)

    agent = Agent(model=model)

    # Create evaluation prompt
    eval_prompt = create_evaluation_prompt(transcript_text, frontmatter)

    # Get judge's assessment
    click.echo("⏳ Requesting evaluation...\n")
    try:
        result = agent.run_sync(eval_prompt)
        evaluation = result.output
    except Exception as e:
        click.echo(f"Error during evaluation: {e}", err=True)
        sys.exit(1)

    # Output results
    if output:
        try:
            output.write_text(evaluation)
            click.echo(f"✅ Evaluation written to: {output}")
        except Exception as e:
            click.echo(f"Error writing output: {e}", err=True)
            sys.exit(1)
    else:
        click.echo("=" * 80)
        wrap_and_echo(evaluation)
        click.echo("=" * 80)


if __name__ == "__main__":
    main()
