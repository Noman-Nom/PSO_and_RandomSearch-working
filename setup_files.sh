#!/bin/bash

echo "Creating all project files..."

# Create empty files
touch models/mlp.py
touch models/cnn.py
touch optimizers/base_optimizer.py
touch optimizers/random_search.py
touch optimizers/pso.py
touch optimizers/bayesian_opt.py
touch utils/trainer.py
touch utils/visualization.py
touch experiments/config.yaml
touch experiments/run_experiment.py
touch experiments/analyze_results.py
touch README.md
touch .gitignore

echo "✅ All files created!"
echo "Now you need to copy the code into each file."

