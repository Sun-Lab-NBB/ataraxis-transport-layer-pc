# Claude Code Instructions

## Session start behavior

At the beginning of each coding session, before making any code changes, you should build a comprehensive understanding
of the codebase by invoking the `/explore-codebase` skill.

This ensures you:
- Understand the project architecture before modifying code
- Follow existing patterns and conventions
- Do not introduce inconsistencies or break integrations

## Style guide compliance

You MUST invoke the appropriate style skill before performing ANY of the following tasks:

| Task                                    | Skill to invoke    |
|-----------------------------------------|--------------------|
| Writing or modifying Python code        | `/python-style`    |
| Writing or modifying README files       | `/readme-style`    |
| Writing or modifying pyproject.toml     | `/pyproject-style` |
| Writing or modifying tox.ini            | `/tox-config`      |
| Writing or modifying Sphinx docs files  | `/api-docs`        |
| Writing git commit messages             | `/commit`          |
| Writing or modifying skill files        | `/skill-design`    |

Each skill contains a verification checklist that you MUST complete before submitting any work. Failure to invoke the
appropriate skill results in style violations.

## Cross-referenced library verification

Ataraxis framework projects often depend on other `ataraxis-*` libraries. These libraries may be stored locally in the
same parent directory as this project (`/home/cyberaxolotl/Desktop/GitHubRepos/`).

**Before writing code that interacts with a cross-referenced library, you MUST:**

1. **Check for local version**: Look for the library in the parent directory (e.g., `../ataraxis-time/`,
   `../ataraxis-base-utilities/`).

2. **Compare versions**: If a local copy exists, compare its version against the latest release or main branch on
   GitHub:
   - Read the local `pyproject.toml` to get the current version
   - Use `gh api repos/Sun-Lab-NBB/{repo-name}/releases/latest` to check the latest release
   - Alternatively, check the main branch version on GitHub

3. **Handle version mismatches**: If the local version differs from the latest release or main branch, notify the user
   with the following options:
   - **Use online version**: Fetch documentation and API details from the GitHub repository
   - **Update local copy**: The user will pull the latest changes locally before proceeding

4. **Proceed with correct source**: Use whichever version the user selects as the authoritative reference for API
   usage, patterns, and documentation.

**Why this matters**: Skills and documentation may reference outdated APIs. Always verify against the actual library
state to prevent integration errors.

## Companion library synchronization

This library (`ataraxis-transport-layer-pc`) and its C++ counterpart (`ataraxis-transport-layer-mc`) implement the
same serial communication protocol on opposite ends of the connection. Any change to the packet format, status codes,
COBS encoding, CRC computation, or buffer layout in this library MUST be synchronized with the corresponding change
in `ataraxis-transport-layer-mc`, and vice versa.

**Before modifying any protocol-level behavior, you MUST:**

1. **Identify the companion repository**: Check for a local copy at `../ataraxis-transport-layer-mc/`. If unavailable,
   use `gh api repos/Sun-Lab-NBB/ataraxis-transport-layer-mc` to access the remote repository.

2. **Review the corresponding implementation**: Read the C++ source that implements the same functionality you are
   modifying. Verify that the current PC and MC implementations are in sync before making changes.

3. **Plan synchronized changes**: Document what must change in both libraries. Notify the user of the required
   companion changes so they can be applied together.

4. **Never modify protocol behavior unilaterally**: A change applied to only one side of the connection will cause
   communication failures. Both libraries must agree on start byte value, delimiter byte value, payload size
   constraints, COBS encoding/decoding logic, CRC polynomial and computation, packet structure and field ordering,
   and status code definitions.

**What requires synchronization:**
- Packet format fields (start byte, delimiter, payload size encoding, CRC postamble)
- COBS encoding/decoding algorithm
- CRC polynomial, initial value, final XOR value, and lookup table generation
- `TransportLayerStatus` code values and meanings
- Buffer size calculations and payload size constraints
- Data serialization byte ordering and type representations

**What does NOT require synchronization:**
- Python-specific wrapper logic, error messages, and `__repr__` formatting
- Test infrastructure (`SerialMock`, pytest fixtures)
- CLI commands (`axtl-ports`)
- Build system, documentation, and packaging

## Available skills

| Skill                    | Description                                                                          |
|--------------------------|--------------------------------------------------------------------------------------|
| `/explore-codebase`      | Perform in-depth codebase exploration at session start                               |
| `/explore-dependencies`  | Explore ataraxis dependency APIs for a live API snapshot                             |
| `/python-style`          | Apply Ataraxis framework Python coding conventions (REQUIRED for all Python changes) |
| `/readme-style`          | Apply Ataraxis framework README conventions (REQUIRED for README changes)            |
| `/pyproject-style`       | Apply Ataraxis framework pyproject.toml conventions                                  |
| `/tox-config`            | Apply Ataraxis framework tox.ini conventions                                         |
| `/api-docs`              | Apply Ataraxis framework API documentation conventions                               |
| `/commit`                | Draft Ataraxis framework style-compliant git commit messages                         |
| `/skill-design`          | Generate and verify skill files and CLAUDE.md project instructions                   |

## Project context

This is **ataraxis-transport-layer-pc**, a Python library for bidirectional serial communication with Arduino and
Teensy microcontrollers over USB and UART interfaces. It is the PC-side counterpart to the companion C++ library
`ataraxis-transport-layer-mc`. The library targets time-critical scientific applications and uses Numba JIT
compilation to achieve microsecond-level communication speeds.

### Key areas

| Directory                          | Purpose                                            |
|------------------------------------|----------------------------------------------------|
| `src/ataraxis_transport_layer_pc/` | Main library source code (2 modules + __init__.py) |
| `tests/`                           | Test suite (pytest with xdist)                     |
| `examples/`                        | Quickstart example script (`rx_tx_loop.py`)        |
| `docs/`                            | Sphinx documentation source                        |
| `envs/`                            | Pre-configured development environment .yml files  |

### Architecture

- **TransportLayer** (`transport_layer.py`): Main class providing the bidirectional serial communication interface.
  Manages dual buffers (transmission and reception), packet construction with COBS encoding and CRC checksums, and
  multi-stage resumable packet parsing. Supports numpy scalars, 1D arrays, and dataclasses as serializable data types.
- **Helper modules** (`helper_modules.py`): Low-level JIT-compiled processing classes. `COBSProcessor` and
  `CRCProcessor` are Python wrappers around Numba `jitclass` instances (`_COBSProcessor`, `_CRCProcessor`) that
  handle Consistent Overhead Byte Stuffing encoding/decoding and CRC-8/16/32 checksum computation. `SerialMock`
  replicates the PySerial `Serial` interface for unit testing without hardware.
- **No MCP server**: This library does not provide an MCP server.

### CLI entry point

| Command      | Entry point                                                         | Purpose                              |
|--------------|---------------------------------------------------------------------|--------------------------------------|
| `axtl-ports` | `ataraxis_transport_layer_pc.transport_layer:print_available_ports` | Display available serial ports       |

### Public API surface

Exported from `__init__.py` via `__all__`:
- `TransportLayer` — Main communication class
- `TransportLayerStatus` — Status code enumeration (IntEnum)
- `COBSProcessor` — COBS encoding/decoding wrapper
- `CRCProcessor` — CRC checksum computation wrapper
- `list_available_ports()` — Returns available serial ports as `ListPortInfo` tuple
- `print_available_ports()` — Prints formatted port list to terminal

### Core components

| Component              | File                 | Purpose                                                     |
|------------------------|----------------------|-------------------------------------------------------------|
| `TransportLayer`       | `transport_layer.py` | Bidirectional serial communication with packet framing      |
| `TransportLayerStatus` | `transport_layer.py` | Status codes for packet parsing and buffer operations       |
| `COBSProcessor`        | `helper_modules.py`  | Python wrapper for JIT-compiled COBS encoder/decoder        |
| `CRCProcessor`         | `helper_modules.py`  | Python wrapper for JIT-compiled CRC checksum calculator     |
| `SerialMock`           | `helper_modules.py`  | Mock serial port replicating PySerial interface for testing |
| `_COBSProcessor`       | `helper_modules.py`  | Numba jitclass for high-performance COBS operations         |
| `_CRCProcessor`        | `helper_modules.py`  | Numba jitclass for high-performance CRC operations          |

### Key patterns

- **Wrapper pattern**: Python classes (`COBSProcessor`, `CRCProcessor`) wrap Numba `jitclass` instances to provide
  Pythonic APIs while preserving JIT compilation benefits. The wrapper handles input validation and error reporting;
  the jitclass handles computation.
- **JIT compilation**: Performance-critical methods use `@njit(cache=True)` or `@jitclass`. First invocation compiles
  to native code (slow); subsequent calls run at C speed. The `# type: ignore[import-untyped]` and
  `# type: ignore[untyped-decorator]` comments on Numba imports and decorators are expected and should not be removed.
- **Status code returns in JIT methods**: JIT-compiled methods return `TransportLayerStatus` enum values instead of
  raising exceptions (Numba limitation). Python wrapper methods convert status codes to exceptions via
  `console.error()`.
- **Resumable packet parsing**: `_parse_packet()` implements a 4-stage state machine that can resume across multiple
  calls when insufficient bytes are available, accumulating partial data in `_leftover_bytes`.
- **SerialMock for testing**: `TransportLayer` accepts `test_mode=True` to substitute `SerialMock` for the real
  `Serial` port, enabling full unit testing without hardware.

### Dependencies

| Library                   | Purpose                                                       |
|---------------------------|---------------------------------------------------------------|
| `numpy`                   | Array operations, serialization, type system                  |
| `numba`                   | JIT compilation for COBS/CRC calculations                     |
| `pyserial`                | Serial port I/O (`Serial` class, `list_ports`)                |
| `ataraxis-time`           | `Timeout` class and `TimerPrecisions` enum for byte reception |
| `ataraxis-base-utilities` | `console` object for unified error handling and output        |

### Code standards

- MyPy strict mode with full type annotations
- Google-style docstrings
- 120 character line limit
- Uses `console.error(message=msg, error=ErrorType)` for all error handling (no bare `raise`)
- See `/python-style` for complete conventions

### Development commands

```bash
tox -e lint              # Format, lint, and type-check
tox -e stubs             # Generate .pyi stub files
tox -e py312-test        # Run tests for Python 3.12
tox -e py313-test        # Run tests for Python 3.13
tox -e py314-test        # Run tests for Python 3.14
tox -e coverage          # Aggregate multi-version coverage reports
tox -e docs              # Build Sphinx API documentation
tox                      # Run full pipeline (uninstall -> export -> lint -> ... -> install)
```

### Workflow guidance

**Modifying TransportLayer:**

1. Review `src/ataraxis_transport_layer_pc/transport_layer.py` for the current implementation
2. Understand the dual-buffer architecture and packet format (start_byte, payload_size, COBS-encoded data, CRC)
3. JIT-compiled methods (`_write_scalar_data`, `_construct_packet`, `_parse_packet`, `_process_packet`) cannot use
   Python objects or raise exceptions — they return status codes
4. Parameters must match the companion `ataraxis-transport-layer-mc` C++ library exactly; mismatches cause
   unrecoverable packet corruption

**Modifying helper modules:**

1. Review `src/ataraxis_transport_layer_pc/helper_modules.py` for existing wrapper/jitclass patterns
2. Numba jitclass instances have strict type requirements — use Numba-compatible types only
3. Python wrappers handle input validation, error reporting via `console.error()`, and `__repr__` formatting
4. The `_COBSProcessor` enforces a 254-byte maximum payload size (COBS protocol hard limit)

**Adding new serializable data types:**

1. Review `write_data()` and `read_data()` in `transport_layer.py` for the current type dispatch logic
2. Supported types: numpy scalars (`uint8` through `float64`, `bool_`), 1D numpy arrays, and dataclasses
3. Dataclass support uses recursive field-by-field serialization via `dataclasses.fields()`
4. New types require corresponding JIT-compiled serialization/deserialization methods

**Important considerations:**

- Max payload size is 254 bytes (COBS protocol hard limit)
- The `# pragma: no cover` annotations on JIT-compiled methods are intentional — Numba JIT code cannot be
  instrumented by coverage tools
- The port close-then-reopen pattern in `__init__` is a Windows workaround for COM port release delays
