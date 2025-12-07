# CLI Tool Domain Template

This template extends the generic templates with command-line tool-specific questions.

## Command Structure
**Questions to ask:**
- What is the main command name?
- What subcommands are needed (e.g., `tool init`, `tool run`, `tool status`)?
- Will you use a single-command or multi-command structure?
- What is the command hierarchy (nested subcommands)?
- Are there aliases or shortcuts for common commands?

**Example answers:**
> - Main command: `paragon-deps`
> - Subcommands: `analyze`, `check`, `graph`, `list`
> - Multi-command structure with clear separation
> - Flat hierarchy (no nested subcommands for v1)
> - Aliases: `ls` for `list`, `viz` for `graph`

---

## Arguments & Options
**Questions to ask:**
- What positional arguments are required for each command?
- What optional flags/options are available?
- What are the short and long forms of options (-v vs --verbose)?
- What are the default values for options?
- Are there mutually exclusive options?
- Are there environment variables that override options?

**Example answers:**
> **Command: `paragon-deps check <service>`**
> - Positional: `service` (required) - Service name to check
> - Options:
>   - `--depth N` / `-d N` - Dependency depth (default: 1)
>   - `--format json|table` / `-f` - Output format (default: table)
>   - `--verbose` / `-v` - Verbose output (default: false)
> - Env vars: `PARAGON_DEPTH` overrides --depth

---

## Input/Output
**Questions to ask:**
- What input does the CLI accept (stdin, files, arguments)?
- What output formats are supported (plain text, JSON, YAML, table)?
- Where does output go (stdout, stderr, files)?
- Will you support piping (command1 | command2)?
- How will you handle interactive input (prompts, confirmations)?
- Will you support progress indicators or spinners?

**Example answers:**
> - Input: Command arguments, optional config file (TOML)
> - Output formats: Table (human), JSON (machine), YAML
> - Output: Results to stdout, errors to stderr, logs to stderr
> - Piping: JSON output designed to be pipeable (e.g., `| jq`)
> - Interactive: Yes, with `--yes` flag to skip confirmations
> - Progress: Spinners for long operations (using Rich library)

---

## Configuration & Settings
**Questions to ask:**
- Will the tool use a configuration file?
- What format (YAML, TOML, JSON, INI)?
- Where will config files be located (~/.config, ./, ./config)?
- What can be configured?
- How are config values merged (CLI args override file)?
- Will you support per-project vs global config?

**Example answers:**
> - Config file: Optional `paragon.toml`
> - Format: TOML (easy to read/write)
> - Location: `~/.config/paragon/config.toml` (global), `./paragon.toml` (project)
> - Configurable: API URL, default format, cache settings
> - Precedence: CLI args > project config > global config > defaults
> - Both per-project and global config supported

---

## Error Handling & Exit Codes
**Questions to ask:**
- What exit codes will you use (0 = success, 1 = error, others)?
- How will errors be displayed (simple message, detailed, JSON)?
- Will you use color coding for errors/warnings?
- How will you handle verbose error modes (stack traces)?
- What happens on invalid arguments or options?

**Example answers:**
> - Exit codes:
>   - 0 = Success
>   - 1 = Conflicts detected (expected failure)
>   - 2 = Invalid arguments/usage error
>   - 3 = System error (API down, network issue)
> - Errors: Red color, prefix with "Error:", detailed message
> - Warnings: Yellow color, prefix with "Warning:"
> - Verbose mode: `--debug` shows stack traces
> - Invalid args: Show usage help and exit code 2

---

## Help & Documentation
**Questions to ask:**
- How will help be displayed (`--help`, `help` subcommand)?
- Will you have man pages or online docs?
- How detailed should help text be?
- Will you show examples in help output?
- How will you document complex workflows (tutorials, guides)?

**Example answers:**
> - Help: `--help` / `-h` flag for any command
> - Man pages: Generate from code using Click autodoc
> - Help detail: Brief description + list of options + examples
> - Examples: Yes, show 2-3 common use cases in help
> - Tutorials: Separate markdown docs + online docs site

---

## Terminal UI & Formatting
**Questions to ask:**
- Will you use colors and formatting (bold, italics)?
- Will you use tables, charts, or other rich formatting?
- How will you handle terminals without color support?
- What terminal width will you assume (or detect)?
- Will you support Unicode characters or stick to ASCII?

**Example answers:**
> - Colors: Yes, using Rich library
> - Rich formatting: Tables, trees, progress bars
> - No color support: Auto-detect, use `NO_COLOR` env var
> - Terminal width: Auto-detect, default to 80 if unavailable
> - Unicode: Yes for check marks (✓), with ASCII fallback

---

## Authentication & Credentials
**Questions to ask:**
- How will users authenticate (API keys, OAuth, login command)?
- Where will credentials be stored (config file, keychain, env vars)?
- How will credentials be provided (flags, prompts, env vars)?
- How will you handle credential expiry/refresh?
- How will you secure credentials (encryption, permissions)?

**Example answers:**
> - Auth: GitLab personal access token
> - Storage: OS keychain (macOS Keychain, Linux Secret Service, Windows Credential Manager)
> - Provision: `paragon-deps login` command (interactive prompt)
> - Env var: `PARAGON_TOKEN` for CI/CD environments
> - Expiry: Show clear error message, prompt to re-login
> - Security: Never log credentials, check file permissions on config

---

## Performance & Caching
**Questions to ask:**
- Will you cache results to improve performance?
- Where will cache be stored (~/.cache, /tmp)?
- What is the cache invalidation strategy (TTL, manual)?
- How will users clear the cache?
- Will you show cache status (hit/miss)?

**Example answers:**
> - Cache: Yes, for analyzed service metadata
> - Location: `~/.cache/paragon/`
> - TTL: 5 minutes for analysis results, 1 hour for service metadata
> - Clear: `paragon-deps cache clear` command
> - Status: Show "Using cached data (2m old)" in verbose mode

---

## Logging & Debugging
**Questions to ask:**
- What logging levels will you support (debug, info, warn, error)?
- Where will logs be written (stderr, file)?
- How do users enable verbose/debug mode?
- Will you have a log file for persistent debugging?
- What information should be logged?

**Example answers:**
> - Levels: ERROR (default), WARN, INFO (`-v`), DEBUG (`-vv`)
> - Output: stderr for console, optional file for persistent logs
> - Enable: `-v` for info, `-vv` for debug, `--log-file=path` for file
> - Log file: `~/.cache/paragon/paragon.log` (rotated, max 10MB)
> - Content: Timestamps, level, message, context (service, operation)

---

## Installation & Distribution
**Questions to ask:**
- How will users install the tool (pip, npm, brew, binary download)?
- What platforms will you support (Linux, macOS, Windows)?
- Will you distribute binaries or require a runtime (Python, Node)?
- How will users update the tool?
- Will you support auto-updates or version checks?

**Example answers:**
> - Installation:
>   - Python: `pip install paragon-deps`
>   - Homebrew: `brew install paragon-deps` (macOS)
>   - Binary: GitHub releases with pre-built binaries
> - Platforms: Linux (x64, ARM), macOS (Intel, Apple Silicon), Windows (x64)
> - Runtime: Python 3.11+ required (or use binary with embedded Python)
> - Updates: `paragon-deps update` or `pip install --upgrade`
> - Version check: Automatic on startup, show message if new version available

---

## Shell Integration
**Questions to ask:**
- Will you provide shell completions (bash, zsh, fish)?
- Will you support shell aliases or functions?
- Will you integrate with shell prompts (e.g., show current service)?
- Do you need shell hooks (preexec, precmd)?

**Example answers:**
> - Completions: Yes, for bash, zsh, fish (auto-generated by Click)
> - Install: `paragon-deps completion install`
> - Aliases: Suggest in docs but don't auto-install
> - No prompt integration needed
> - No shell hooks needed

---

## CI/CD Integration
**Questions to ask:**
- Will this tool run in CI/CD environments?
- How will it handle non-interactive environments?
- What output format is best for CI/CD (JSON, exit codes)?
- Will you provide GitHub Actions / GitLab CI templates?
- How will authentication work in CI/CD?

**Example answers:**
> - CI/CD: Yes, designed to run in pipelines
> - Non-interactive: Auto-detect TTY, no prompts if not a TTY
> - CI output: `--format=json` for machine parsing
> - Templates: Provide `.gitlab-ci.yml` example
> - Auth: Use `PARAGON_TOKEN` env var (no keychain in CI)

---

## Plugin/Extension System
**Questions to ask:**
- Will you support plugins or extensions?
- How will plugins be discovered and loaded?
- What capabilities can plugins add (subcommands, formatters)?
- How will plugins be installed (pip, manual)?

**Example answers:**
> - Plugins: Not in v1 (future consideration)
> - Future: Entry point-based plugins (setuptools entry_points)
> - Capabilities: Custom analyzers for additional languages
> - Installation: `pip install paragon-deps-plugin-go`

---

## Compatibility & Versioning
**Questions to ask:**
- What is the versioning scheme (SemVer)?
- How will you handle breaking changes?
- Will you support multiple major versions concurrently?
- How will you deprecate features?
- What is the minimum supported version of dependencies (Python, etc.)?

**Example answers:**
> - Versioning: Semantic Versioning (SemVer 2.0)
> - Breaking changes: Major version bump, migration guide
> - Concurrent versions: Support latest major + previous major
> - Deprecation: Warning in output for 1 major version, remove in next
> - Minimum: Python 3.11+, rustworkx 0.14+

---

## Testing Strategy
**Questions to ask:**
- How will you test CLI commands (unit, integration, e2e)?
- How will you test interactive prompts?
- How will you test output formatting?
- How will you test on different platforms?
- How will you test shell completions?

**Example answers:**
> - Unit: Test individual functions with pytest
> - Integration: Test full commands with Click CliRunner
> - E2E: Shell scripts that run actual CLI in subprocess
> - Prompts: Mock input using CliRunner.invoke(input="...")
> - Output: Snapshot testing for formatted output
> - Platforms: GitHub Actions matrix (Linux, macOS, Windows)
> - Completions: Manual testing (hard to automate)

---

## User Experience
**Questions to ask:**
- How will you make the tool beginner-friendly (good errors, examples)?
- What are the most common workflows to optimize?
- Will you provide a getting-started tutorial?
- How will you handle typos or similar commands (suggestions)?
- Will you track usage analytics (with opt-in)?

**Example answers:**
> - Beginner-friendly: Clear error messages with suggestions, `--help` everywhere
> - Common workflows: Check before deploy (optimize for speed)
> - Tutorial: `paragon-deps quickstart` command that runs interactive setup
> - Typo handling: Suggest similar commands (e.g., "Did you mean 'analyze'?")
> - Analytics: No usage tracking (privacy-first)
