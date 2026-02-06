# credit_risk_example

MRM project for model risk management.

## Getting Started

```bash
# List models
mrm list models

# Run validation tests
mrm test --models my_model

# Generate documentation
mrm docs generate
```

## Project Structure

- `models/` - Model definitions (YAML files)
- `tests/custom/` - Custom test implementations
- `data/` - Training and validation datasets
- `docs/` - Documentation templates

## Adding a Model

1. Create model definition in `models/`
2. Define tests in the YAML file
3. Run tests: `mrm test --models model_name`
