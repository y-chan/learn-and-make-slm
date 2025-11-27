# AGENTS.md

This file provides guidance to Coding Agents when working with code in this repository.

YOU MUST CHECK `mise run check` passing before finishing any task.

## Project Overview

Transformerとその基礎モジュールを工学的に学び実装する教育プロジェクト。
最終的に動作するSmall Language Modelの構築を目指す。

## Architecture

### Module Structure

```
models/
├── basic/           # 基礎的なニューラルネットワークコンポーネント
└── transformer/     # Transformer固有のコンポーネント
```

### Key Design Decisions

- **Custom implementations**: PyTorchの組み込み実装を使わず、各コンポーネントを自前で実装
- **Type safety**: jaxtyping + beartype による実行時型チェック
- **Test-driven**: 各モジュールに対応するテストファイルを配置
  - `models/basic/linear.py` → `tests/basic/linear_test.py`

### Type Annotations

Typing tensors with `jaxtyping` for better clarity and safety.

<https://docs.kidger.site/jaxtyping/api/array/>

## Development Commands

### Environment Setup

```bash
# Install dependencies
uv sync
```

### Development Workflow

```bash
# Run all checks (lint, format, test)
mise run check

# Individual commands
mise run lint:fix  # Auto-fix linting issues
mise run fmt       # Format code with ruff
mise run test      # Run pytest

# Run specific test file
mise run test tests/basic/linear_test.py
mise run test tests/transformer/rope_test.py
```

### Important Rules

- **MUST** run `mise run check` and ensure it passes before completing any task
- Python 3.13+ required
- ruff configured with line-length=125, ignores F722 for jaxtyping

## Testing Strategy

- Tests use PyTorch's built-in implementations as ground truth for validation
- Each module in `models/` should have corresponding test in `tests/`
- Test structure mirrors source structure: `models/basic/linear.py` → `tests/basic/linear_test.py`
- `tests/conftest.py` adds project root to sys.path for imports

## Running Individual Modules

各モジュールは個別に動作確認可能な設計 (READMEより)
