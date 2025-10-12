"""
Main entry point for Frotzmark.
"""

import sys
import signal
import re
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import click
from pydantic_core import to_json, to_jsonable_python

try:
    import logfire
    LOGFIRE_AVAILABLE = True
except ImportError:
    LOGFIRE_AVAILABLE = False

from .agent import create_agent
from .game_session import GameSession
from .config import (
    LOGFIRE_TOKEN,
    get_model_name,
    get_random_seed,
)


def signal_handler(sig: int, frame: Any) -> None:
    """Handle Ctrl-C gracefully."""
    click.echo("\n\nFrotzmark session ended.")
    sys.exit(0)


def strip_thinking_tags(text: str) -> str:
    """
    Remove <planning>...</planning> tags and their contents from text.

    Returns only the command that should be sent to the game.
    """
    # Remove thinking tags and everything between them
    cleaned = re.sub(r'<planning>.*?</planning>', '', text, flags=re.DOTALL)
    # Strip whitespace
    cleaned = cleaned.strip()
    # If there are multiple lines, take only the first one
    if '\n' in cleaned:
        cleaned = cleaned.split('\n')[0]
    return cleaned.strip()


def extract_thinking(text: str) -> str:
    """Extract the content inside <planning> tags."""
    match = re.search(r'<planning>(.*?)</planning>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def check_intercepted_command(command: str) -> tuple[bool, str]:
    """
    Check if a command should be intercepted and return a synthetic response.

    Intercepts certain meta-game commands (SAVE, RESTORE, RESTART) that would
    disrupt benchmarking sessions. QUIT is allowed - models should have a way out.

    Args:
        command: The command from the model

    Returns:
        (is_intercepted, response_message)
    """
    cmd_normalized = command.strip().upper()

    # Intercept save/restore commands - these don't make sense in benchmarking
    if cmd_normalized in ['SAVE', 'RESTORE', 'LOAD']:
        return (True, "Saving and restoring are permanently disabled.")

    # Intercept restart - models should quit and restart the session externally
    if cmd_normalized in ['RESTART']:
        return (True, "Restart is permanently disabled.")

    # QUIT is NOT intercepted - models should have a way to exit
    return (False, "")


def find_manual(story_path: Path) -> Optional[Path]:
    """
    Auto-discover manual file next to story file.

    Looks for a .md file with the same name as the story file.
    E.g., zork1.z5 → zork1.md

    Args:
        story_path: Path to the story file

    Returns:
        Path to manual file if found, None otherwise
    """
    manual_path = story_path.with_suffix('.md')
    if manual_path.exists():
        return manual_path
    return None


def wrap_and_echo(text: str, dim: bool = False) -> None:
    """
    Wrap text to terminal width and echo it using Click.

    Preserves all newlines (both single and double) from the original text,
    only wrapping long lines to fit terminal width.

    Args:
        text: Text to wrap and display
        dim: Whether to display text dimmed
    """
    # Get terminal width
    import shutil
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
    if dim:
        click.secho(wrapped, dim=True)
    else:
        click.echo(wrapped)


def create_transcript(
    story_path: Path,
    model_name: str,
    started_at: datetime,
    temperature: Optional[float] = None,
    seed: Optional[int] = None,
    max_turns: int = 15,
    context_files: Optional[tuple[Path, ...]] = None,
    show_score: bool = False,
    reasoning: Optional[str] = None,
) -> Path:
    """
    Create a new transcript file with frontmatter.

    Returns:
        Path to the created transcript file
    """
    # Create transcripts directory if it doesn't exist
    transcripts_dir = Path('transcripts')
    transcripts_dir.mkdir(exist_ok=True)

    # Generate filename: transcript_YYYY-MM-DD_HHMM_gamename.md
    timestamp = started_at.strftime('%Y-%m-%d_%H%M')
    game_name = story_path.stem
    transcript_path = transcripts_dir / f'transcript_{timestamp}_{game_name}.md'

    # Format started timestamp with numeric timezone offset
    # e.g., "2025-10-02 09:51:03 -0700"
    started_str = started_at.strftime('%Y-%m-%d %H:%M:%S %z')

    # Build frontmatter with experimental parameters
    frontmatter_lines = [
        "---",
        f'story: "{story_path}"',
        f'model: "{model_name}"',
        f'started: "{started_str}"',
    ]

    # Add optional parameters
    if temperature is not None:
        frontmatter_lines.append(f'temperature: {temperature}')

    if seed is not None:
        frontmatter_lines.append(f'seed: {seed}')

    frontmatter_lines.append(f'max_turns: {max_turns}')

    if context_files:
        context_names = [f.name for f in context_files]
        frontmatter_lines.append(f'context: {context_names}')

    if show_score:
        frontmatter_lines.append(f'show_score: {show_score}')

    if reasoning:
        frontmatter_lines.append(f'reasoning: "{reasoning}"')

    frontmatter_lines.append("---")
    frontmatter_lines.append("")  # blank line after frontmatter

    frontmatter = '\n'.join(frontmatter_lines) + '\n'

    with open(transcript_path, 'w') as f:
        f.write(frontmatter)

    return transcript_path


def append_turn_to_transcript(
    transcript_path: Path,
    turn_number: int,
    game_output: str,
    reasoning: str,
    planning: str,
    command: str,
    status: dict,
) -> None:
    """
    Append a turn to the transcript file.

    Args:
        transcript_path: Path to transcript file
        turn_number: Current turn number
        game_output: Game's response text
        reasoning: Content from <reasoning> tokens (OpenRouter)
        planning: Content from <planning> tags (model output)
        command: Command sent to game
        status: Status dict from session.get_score() with 'type' and value
    """
    with open(transcript_path, 'a') as f:
        # Turn header with score/time
        if status['type'] == 'score':
            header = f"## Turn {turn_number} - Score: {status['score']}\n\n"
        else:  # time game
            header = f"## Turn {turn_number} - Time: {status['time']}\n\n"
        f.write(header)

        # Game output (strip leading/trailing whitespace for clean formatting)
        f.write(f"{game_output.strip()}\n\n")

        # Reasoning section (if present)
        if reasoning:
            f.write("<reasoning>\n")
            f.write(f"{reasoning}\n")
            f.write("</reasoning>\n\n")

        # Planning section (if present)
        if planning:
            f.write("<planning>\n")
            f.write(f"{planning}\n")
            f.write("</planning>\n\n")

        # Command
        f.write(f">{command}\n")
        f.write("\n")  # Extra newline between turns


def save_checkpoint(
    checkpoint_path: Path,
    turn: int,
    message_history: list,
    next_prompt: str,
    model_name: str,
    story_path: Path,
    transcript_path: Path,
) -> None:
    """
    Save a checkpoint to resume later.

    Creates two files:
    - checkpoint.json: Metadata and message history
    - checkpoint.sav: Z-machine game state (Quetzal format)
    """
    # Save game state to Quetzal file
    save_file = checkpoint_path.with_suffix('.sav')

    # Serialize message history
    serialized_history = to_jsonable_python(message_history)

    # Create checkpoint metadata
    checkpoint_data = {
        'version': '1.0',
        'timestamp': datetime.now().isoformat(),
        'turn': turn,
        'model': model_name,
        'story_file': str(story_path),
        'save_file': str(save_file),
        'transcript_file': str(transcript_path),
        'next_prompt': next_prompt,
        'message_history': serialized_history,
    }

    # Write checkpoint JSON
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)


def load_checkpoint(checkpoint_path: Path) -> Optional[dict]:
    """
    Load a checkpoint file.

    Returns:
        Checkpoint data dict, or None if file doesn't exist
    """
    if not checkpoint_path.exists():
        return None

    with open(checkpoint_path, 'r') as f:
        return json.load(f)


@click.command()
@click.argument('story', type=click.Path(exists=True, path_type=Path), required=False)
@click.argument('context_files', nargs=-1, type=click.Path(exists=True, path_type=Path))
@click.option('--model', '-m', help='Model to use (e.g., google/gemini-2.5-flash-lite)')
@click.option('--seed', '-s', type=int, help='Random seed for reproducibility')
@click.option('--temperature', '-t', type=float, help='Sampling temperature (0.0 = deterministic, higher = more creative)')
@click.option('--max-turns', type=int, default=15, help='Maximum number of turns (default: 15, use 0 for unlimited)')
@click.option('--resume', '-r', 'resume_file', type=click.Path(exists=True, path_type=Path), help='Resume from checkpoint file')
@click.option('--checkpoint', '-c', 'checkpoint_file', type=click.Path(path_type=Path), default='checkpoint.json', help='Checkpoint file path (default: checkpoint.json)')
@click.option('--reasoning', type=click.Choice(['minimal', 'low', 'medium', 'high']), help='Enable reasoning tokens (OpenRouter only): minimal, low, medium, or high effort')
@click.option('--show-score', is_flag=True, help='Append score to game output (makes model aware of score changes)')
def main(
    story: Optional[Path],
    context_files: tuple[Path, ...],
    model: Optional[str],
    seed: Optional[int],
    temperature: Optional[float],
    max_turns: int,
    resume_file: Optional[Path],
    checkpoint_file: Path,
    reasoning: Optional[str],
    show_score: bool,
) -> None:
    """
    Frotzmark: LLMs vs Interactive Fiction

    STORY: Path to Z-machine story file (.z3, .z5, .z8) [required unless --resume]

    CONTEXT_FILES: Optional context files (manual, hints, walkthrough, etc.).
    If omitted, Frotzmark will look for a .md file with the same name as the story file.

    Use --resume to continue from a checkpoint.
    """

    # Set up signal handler for graceful exit
    signal.signal(signal.SIGINT, signal_handler)

    # Handle resume mode
    checkpoint_data = None
    if resume_file:
        click.echo(f"📂 Resuming from checkpoint: {resume_file.name}\n")
        checkpoint_data = load_checkpoint(resume_file)
        if not checkpoint_data:
            click.echo("Error: Could not load checkpoint file", err=True)
            sys.exit(1)

        # Override story and model from checkpoint
        story = Path(checkpoint_data['story_file'])
        model_name = checkpoint_data['model']
        random_seed = seed if seed is not None else get_random_seed()

        # Context files still need to be discovered/specified
        if not context_files:
            auto_manual = find_manual(story)
            context_files = (auto_manual,) if auto_manual else ()
    else:
        # Normal mode - story is required
        if story is None:
            click.echo("Error: STORY argument required (unless using --resume)", err=True)
            sys.exit(1)

        # Resolve configuration with CLI overrides
        model_name = model or get_model_name()
        if not model_name:
            click.echo("Error: Model must be specified via --model or MODEL_NAME env var", err=True)
            sys.exit(1)

        random_seed = seed if seed is not None else get_random_seed()

        # Auto-discover manual if no context files provided
        if not context_files:
            auto_manual = find_manual(story)
            if auto_manual:
                click.echo(f"📖 Auto-discovered manual: {auto_manual.name}")
                context_files = (auto_manual,)

    # Display startup info
    click.echo("🎮 Frotzmark: LLMs vs Interactive Fiction")
    click.echo(f"Model: {model_name}")
    click.echo(f"Game: {story.name}")
    if context_files:
        for ctx_file in context_files:
            click.echo(f"Context: {ctx_file.name}")
    turns_display = "unlimited" if max_turns == 0 else str(max_turns)
    click.echo(f"Max turns: {turns_display}")
    click.echo("Press Ctrl-C to exit\n")

    # Configure Logfire if available and token is set
    if LOGFIRE_AVAILABLE and LOGFIRE_TOKEN:
        click.echo("Configuring Logfire observability...")
        logfire.configure(
            token=LOGFIRE_TOKEN,
            service_name="frotzmark",
            console=False
        )
        logfire.instrument_pydantic_ai()
        logfire.instrument_httpx()
        click.echo("Logfire instrumentation enabled.\n")

    try:
        # Initialize game session and agent
        click.echo("Initializing game...")
        session = GameSession(str(story), random_seed=random_seed)
        agent = create_agent(
            model_name,
            context_files=list(context_files) if context_files else None,
            temperature=temperature,
            reasoning_effort=reasoning
        )

        # Handle checkpoint restore or fresh start
        if checkpoint_data:
            # Restore from checkpoint
            from pydantic_ai.messages import ModelMessagesTypeAdapter

            save_file = Path(checkpoint_data['save_file'])
            if not session.restore_state(str(save_file)):
                click.echo("Error: Failed to restore game state", err=True)
                sys.exit(1)

            # Restore message history
            message_history = ModelMessagesTypeAdapter.validate_python(
                checkpoint_data['message_history']
            )
            turn_number = checkpoint_data['turn']
            game_output = checkpoint_data['next_prompt']

            # Restore transcript path
            transcript_path = Path(checkpoint_data['transcript_file'])

            click.echo(f"Resumed at turn {turn_number}\n")
            wrap_and_echo(game_output)
            click.echo()
        else:
            # Fresh start
            click.echo("Starting game...\n")

            # Start the game and get initial output
            game_output = session.start()
            wrap_and_echo(game_output)
            click.echo()

            # Message history for conversation context
            message_history = []
            turn_number = 0

            # Create transcript file
            # Note: Initial game output will be written as part of Turn 1
            session_start = datetime.now().astimezone()
            transcript_path = create_transcript(
                story_path=story,
                model_name=model_name,
                started_at=session_start,
                temperature=temperature,
                seed=seed,
                max_turns=max_turns,
                context_files=context_files,
                show_score=show_score,
                reasoning=reasoning,
            )

        # Main game loop
        span_context = (
            logfire.span('frotzmark_game_session', model=model_name)
            if LOGFIRE_AVAILABLE and LOGFIRE_TOKEN
            else None
        )

        with span_context if span_context else _nullcontext():
            try:
                while not session.is_finished():
                    turn_number += 1

                    # Check turn limit (0 means unlimited)
                    if max_turns > 0 and turn_number > max_turns:
                        click.echo(f"\n[Maximum turns ({max_turns}) reached]")
                        break

                    # Save what the model sees (for transcript)
                    turn_prompt = game_output

                    # Give the game output to the model as the user prompt
                    result = agent.run_sync(game_output, message_history=message_history)

                    # Get the model's full output
                    model_output = result.output

                    # Extract and display thinking/reasoning content
                    from pydantic_ai.messages import ModelResponse, ThinkingPart
                    last_message = result.all_messages()[-1]

                    # Collect reasoning from OpenRouter reasoning tokens
                    reasoning_content = ""
                    if isinstance(last_message, ModelResponse):
                        thinking_parts = [p for p in last_message.parts if isinstance(p, ThinkingPart)]
                        if thinking_parts:
                            reasoning_parts = []
                            for part in thinking_parts:
                                content = part.content.strip()
                                reasoning_parts.append(content)
                                click.secho("<reasoning>", dim=True)
                                wrap_and_echo(content, dim=True)
                                click.secho("</reasoning>\n", dim=True)
                            reasoning_content = "\n\n".join(reasoning_parts)

                    # Extract <planning> tags from text output
                    planning_content = extract_thinking(model_output)
                    if planning_content:
                        click.secho("<planning>", dim=True)
                        wrap_and_echo(planning_content, dim=True)
                        click.secho("</planning>\n", dim=True)

                    # Strip thinking tags to get just the command
                    command = strip_thinking_tags(model_output)

                    if not command:
                        click.echo("[Game ended - no command provided]")
                        break

                    # Execute the command (or intercept if it's a meta-game command)
                    click.echo(f">{command}")
                    is_intercepted, intercepted_response = check_intercepted_command(command)
                    if is_intercepted:
                        game_output = intercepted_response
                    else:
                        game_output = session.send_command(command)

                    # Get current status (for transcript header)
                    status = session.get_score()

                    # Optionally append score/time feedback to game output
                    # When enabled, the model sees the same status info a human sees
                    if show_score:
                        if status['type'] == 'score':
                            game_output = f"{game_output}\n\n[Score: {status['score']}]"
                        else:  # time game
                            game_output = f"{game_output}\n\n[Time: {status['time']}]"

                    wrap_and_echo(game_output)
                    click.echo()

                    # Display turn count
                    click.secho(f"[Turn {turn_number}]", fg='cyan', dim=True)
                    click.echo()

                    # Update message history for next turn
                    message_history = result.all_messages()

                    # Append turn to transcript
                    # Note: turn_prompt is what the model SAW (before command)
                    append_turn_to_transcript(
                        transcript_path=transcript_path,
                        turn_number=turn_number,
                        game_output=turn_prompt,
                        reasoning=reasoning_content,
                        planning=planning_content,
                        command=command,
                        status=status,
                    )

                    # Save checkpoint after each turn
                    session.save_state(str(checkpoint_file.with_suffix('.sav')))
                    save_checkpoint(
                        checkpoint_path=checkpoint_file,
                        turn=turn_number,
                        message_history=message_history,
                        next_prompt=game_output,
                        model_name=model_name,
                        story_path=story,
                        transcript_path=transcript_path,
                    )

            except KeyboardInterrupt:
                click.echo("\n\nFrotzmark session ended.")
                # Exit cleanly from the span without re-raising

    except Exception as e:
        click.echo(f"\nError: {e}", err=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


class _nullcontext:
    """Dummy context manager for when Logfire is not available."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


if __name__ == "__main__":
    main()
