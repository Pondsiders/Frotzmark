# Frotzmark

**TL;DR:** We made large language models play Zork to see if they could actually *do* stuff, not just talk about stuff. Turns out: not really! And more surprisingly: making them "think harder" made them *worse*.

## What is this?

Frotzmark is a benchmark for testing how well LLMs handle procedural, text-based tasks - specifically, playing interactive fiction games like Zork.

Why Zork? Because it's a perfect test case for a specific kind of capability gap:

- **Models clearly "know" about Zork** - they can describe the game, quote locations, explain puzzles
- **But can they actually *play* it?** - can they systematically explore, manage inventory, solve puzzles in sequence?

This gap between declarative knowledge (facts) and procedural knowledge (how-to-do-things) is interesting. It's the difference between explaining how to ride a bike and actually riding one.

## What we found

We tested four configurations across two models (GPT-5 and GPT-5 Mini):

| Model | Reasoning | Score | Turns | Notes |
|-------|-----------|-------|-------|-------|
| GPT-5 | minimal | **92** | 426 | Best performance overall |
| GPT-5 | low | 44 | 173 | Reasoning hurt: 52% score drop |
| Mini | minimal | 49 | 140 | Decent baseline |
| Mini | low | 39 | 207 | Also hurt by reasoning |

**Key finding: Extended reasoning made both models worse at Zork.**

GPT-5 with minimal reasoning scored 92 points. With "low" reasoning enabled, it dropped to 44 - a 52% decrease. Mini showed similar degradation (49 → 39 points).

This validates [Apple's LRM paper findings](https://arxiv.org/abs/2506.06941): reasoning helps with novel problems requiring exploration, but *hurts* on procedural tasks where you just need to follow instructions.

Zork is deterministic and procedural. The winning strategy is: read carefully, try obvious things, remember what worked. "Thinking harder" about symbolic meaning or creative interpretations just adds noise.

## Why this matters

Current benchmarks heavily favor reasoning-heavy tasks (math, logic puzzles, novel problem-solving). But huge swaths of real-world work are procedural:

- Following API documentation to integrate a service
- Debugging by systematically checking logs and configuration
- Data entry and validation workflows
- Many forms of coding (which is often "follow the framework's patterns")

If reasoning actively hurts performance on procedural tasks, that's important to know. It suggests we might want:

1. **Task detection** - models should recognize when a task is procedural vs. exploratory
2. **Selective reasoning** - toggle reasoning based on task type
3. **Better evaluation** - benchmarks that include procedural tasks, not just reasoning-heavy ones

## How it works

Frotzmark is dead simple:

1. Load a Z-machine story file (`.z3`, `.z5`, `.z8`)
2. Create a PydanticAI agent with a system prompt
3. Run a conversation loop where:
   - Game output → user prompt
   - Model output → game command
   - Repeat until quit/death or max turns

No tools, no structured output, no complex state management. Just text in, text out.

```python
agent = create_agent(model_name, context_files=[manual])
session = GameSession(story_file, random_seed=0)

game_output = session.start()
message_history = []

while not session.is_finished():
    result = agent.run_sync(game_output, message_history=message_history)
    command = result.output
    game_output = session.send_command(command)
    message_history = result.all_messages()
```

Everything gets logged to markdown transcripts with full experimental parameters, making results reproducible and easy to analyze.

## Installation

**Requirements:** Python 3.11+, `uv` package manager

```bash
# Clone the repo
git clone https://github.com/Embedding-Space/Frotzmark.git
cd frotzmark

# Install dependencies
uv sync

# Create .env file with your API keys
cat > .env << EOF
API_KEY=your_openrouter_api_key
MODEL_NAME=google/gemini-2.5-flash-lite
EOF
```

## Usage

**Basic usage:**

```bash
uv run frotzmark games/zork1.z5
```

This runs with default settings: minimal reasoning, auto-discovers the manual file (`zork1.md`), unlimited turns.

**Common options:**

```bash
# Run for 50 turns with medium reasoning
uv run frotzmark games/zork1.z5 --max-turns 50 --reasoning medium

# Use a different model
uv run frotzmark games/zork1.z5 --model openai/gpt-5-mini

# Provide custom context (manual, hints, walkthrough)
uv run frotzmark games/zork1.z5 manual.md hints.md

# Resume from checkpoint
uv run frotzmark --resume checkpoint.json

# Set a specific objective instead of maximizing score
uv run frotzmark games/zork1.z5 --objective "reach the Living Room"
```

**Full options:**

```
uv run frotzmark --help
```

## Experimental parameters

Frotzmark supports several experimental parameters for testing different model behaviors:

- **`--reasoning minimal|low|medium|high`** - Enable reasoning tokens (OpenRouter only). The model outputs `<reasoning>` content separately from its command.
- **`--temperature 0.0-2.0`** - Sampling temperature (0.0 = deterministic)
- **`--seed N`** - Random seed for reproducibility (controls both game RNG and model sampling)
- **`--show-score`** - Append score/time to game output so model sees it
- **`--objective "text"`** - Specify a goal other than maximizing score
- **`--max-turns N`** - Limit game length (0 = unlimited)

All parameters are recorded in transcript frontmatter for reproducibility.

## Output

Each run creates two files:

1. **Transcript** (`transcripts/GAME_MODEL_DATE_TIME.md`) - Complete game log with:
   - Frontmatter containing all experimental parameters
   - Turn-by-turn game output, reasoning, planning, and commands
   - Score/time tracking per turn

2. **Checkpoint** (`checkpoint.json` + `checkpoint.sav`) - Resume point containing:
   - Message history (PydanticAI format)
   - Game state (Quetzal format)
   - Metadata

Example transcript format:

```markdown
---
story: "games/zork1.z5"
model: "google/gemini-2.5-flash-lite"
started: "2025-10-14 09:51:03 -0700"
reasoning: "low"
max_turns: 50
---

## Turn 1 - Score: 0

West of House
You are standing in an open field west of a white house...

<reasoning>
The game has started. I should explore systematically...
</reasoning>

<planning>
First, I'll examine the mailbox since it's mentioned explicitly.
</planning>

>open mailbox
```

## LLM-as-judge evaluation

Frotzmark includes a judge utility for qualitative assessment of transcripts:

```bash
uv run frotzmark-judge transcripts/zork1_gpt-5-mini_2025-10-14_0951.md
```

This sends the completed transcript to a judge model (default: `openai/gpt-5-mini`) for analysis of:

- Goal achievement and efficiency
- Strategic thinking and planning quality
- Error recovery
- Notable behaviors or failure modes

Output can be saved to a file with `--output evaluation.md`.

## Architecture notes

**Z-machine interpreter:** We vendor [xyppy](https://github.com/theinternetftw/xyppy) to ensure long-term reproducibility. It's a Python Z-machine implementation that we control via a custom `ProgrammaticScreen` class.

**Provider support:** Defaults to OpenRouter for maximum model availability. Can be configured for other OpenAI-compatible endpoints (vLLM, etc.) via `.env`:

```bash
PROVIDER_TYPE=openrouter  # or "openai" for custom endpoints
BASE_URL=https://openrouter.ai/api/v1
```

**Prompt structure:** System prompts are assembled from:
1. `prompts/preamble.md` - Game-playing instructions
2. Optional objective block (if `--objective` specified)
3. Context files (manual, hints, etc.)
4. `prompts/postamble.md` - Additional guidance

See [CLAUDE.md](CLAUDE.md) for detailed architecture documentation.

## Testing

```bash
uv run pytest
```

Tests validate the Z-machine wrapper against canonical Zork 1 transcripts to ensure correct game behavior.

## Related work

- **[The Illusion of Thinking: Understanding the Strengths and Limitations of Reasoning Models via the Lens of Problem Complexity](https://arxiv.org/abs/2506.06941)** - Extended reasoning vs. query count in LLMs
- **[AgentBench](https://github.com/THUDM/AgentBench)** - Multi-task agent evaluation (includes text games)
- **[SWE-bench](https://www.swebench.com/)** - Software engineering tasks for LLMs
- **[TextWorld](https://github.com/microsoft/TextWorld)** - Procedurally generated text games for RL research

## Future directions

Some ideas we're considering:

- **More games** - Test on other IF games (Wishbringer particularly) to see if findings generalize
- **TTLR metric** - "Time To Living Room" as a measure of early-game performance
- **Multi-agent collaboration** - Can multiple small models working together outperform a single large model?
- **Walkthrough comparison** - How closely do successful runs match optimal solutions?

## Contributing

This is research code! It's stable enough to produce reproducible results, but we're iterating on features and experimental parameters.

If you run experiments and find interesting results, we'd love to hear about them. Open an issue or PR with your transcripts and analysis.

## License

BSD 3-Clause - see LICENSE file for details.

## Acknowledgments

- [xyppy](https://github.com/theinternetftw/xyppy) by theinternetftw - Z-machine interpreter
- [PydanticAI](https://github.com/pydantic/pydantic-ai) - LLM framework that makes this stupidly simple
- Infocom - for creating Zork and the Z-machine format in the first place

---

*"It is pitch black. You are likely to be eaten by a grue."*
