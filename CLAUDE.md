# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

# Frotzmark

**10/1/2025**

Frotzmark pits large language models against vintage computer games to illuminate gaps between learned knowledge and apply-able knowledge.

(We're working on the language.)

Here's the program in pseudocode:

intro = game.initialize()
output = model.run(intro)
print(output)
BEGIN LOOP
    result = game.run(output)
    print(result)
    output = model.run(result)
END LOOP

## Development Commands

**Install dependencies:**
```bash
uv sync
```

**Run Frotzmark (requires .env with API_KEY and MODEL_NAME):**
```bash
uv run frotzmark games/zork1.z5
```

**Run tests:**
```bash
uv run pytest
```

**Run LLM-as-judge evaluation on a transcript:**
```bash
uv run frotzmark-judge transcripts/zork1_gpt-5-mini_2025-10-14_0951.md
```

## Architecture

### Core Abstractions

**GameSession** ([frotzmark/game_session.py](frotzmark/game_session.py:24)) wraps the vendored Z-machine interpreter (xyppy) and provides:
- `start()` → initial game output
- `send_command(cmd)` → game response
- `get_score()` → current score/moves or time (depending on game type)
- `save_state()` / `restore_state()` → checkpoint management (Quetzal format)

**ProgrammaticScreen** ([frotzmark/screen.py](frotzmark/screen.py:11)) replaces xyppy's terminal-based I/O:
- Captures game output to a buffer instead of stdout
- Pulls commands from a queue instead of stdin
- Raises `StopIteration` when waiting for input (allows turn-by-turn execution)

**Agent creation** ([frotzmark/agent.py](frotzmark/agent.py:70)) assembles system prompts from:
1. `prompts/preamble.md` (game-playing instructions)
2. Optional objective block (injected if `--objective` specified)
3. Context files (manual, hints, etc.)
4. `prompts/postamble.md` (additional guidance)

### Main Loop

The main loop ([frotzmark/__main__.py](frotzmark/__main__.py:485-593)) does:
1. Give game output to agent as user prompt
2. Extract command from agent's response (stripping `<planning>` tags)
3. Send command to game
4. Capture game output
5. Append turn to transcript
6. Save checkpoint
7. Repeat until quit/death or max turns

Transcripts are written to `transcripts/` with frontmatter containing all experimental parameters (model, temperature, reasoning effort, objective, etc.).

### Key Design Decisions

**Conversation-as-game-loop**: Game output becomes the user prompt, model output becomes the game command. No tools, no structured output - just text in, text out. PydanticAI manages message history.

**Turn-level checkpointing**: After every turn, we save both:
- `checkpoint.json` (metadata + PydanticAI message history)
- `checkpoint.sav` (Z-machine state in Quetzal format)

This allows resuming mid-game with `--resume checkpoint.json`.

**Reasoning tokens**: OpenRouter supports `--reasoning minimal|low|medium|high`, which adds `<reasoning>` tokens to model output (separate from the text command). These appear as `ThinkingPart` in PydanticAI's message history.

**Planning tags**: Models can output `<planning>text</planning>` in their response. This is stripped before sending to the game, but preserved in transcripts.

## Technical details

- Use `uv` exclusively
- Use vendored `xyppy` to run Z-machine games (see Dependencies below)
- Use PydanticAI to abstract away all the LLM interface stuff

## Dependencies

### Vendored Code

This project vendors [xyppy](https://github.com/theinternetftw/xyppy), a Python Z-machine interpreter, located in `frotzmark/vendor/xyppy/`. We vendor it (rather than using it as a dependency) to ensure long-term reproducibility of research results. See `frotzmark/vendor/README.md` for details.

**Important**: Don't import xyppy directly - use it through GameSession. The vendoring setup adds it to sys.path in [game_session.py](frotzmark/game_session.py:15-17).

## How it works (from Zorkmark)

PydanticAI makes this stupid simple:

1. **Create an agent** with a system prompt and model config:
   ```python
   from pydantic_ai import Agent
   from pydantic_ai.models.openai import OpenAIChatModel

   agent = Agent(
       model=OpenAIChatModel('some-model'),
       system_prompt="You are playing Zork..."
   )
   ```

2. **Run the game loop** - just pass game output as the user prompt:
   ```python
   message_history = []
   game_output = game.initial_text

   while True:
       # Model sees game output as user message
       result = agent.run_sync(game_output, message_history=message_history)

       # Model's response is the command
       command = result.output

       # Execute command, get new game output
       game_output = game.send_command(command)

       # Update history for next turn
       message_history = result.all_messages()
   ```

That's it. PydanticAI handles:
- Message history management
- Provider abstraction (OpenAI, Anthropic, etc.)
- Serialization for checkpointing
- All the async/streaming complexity

The game output literally becomes the user prompt. The model's output literally becomes the game command. No tools, no complex state management, just a conversation.
