#!/bin/bash

# Script to properly push all files to GitHub

echo "=========================================="
echo "PUSHING COMPLETE FRAMEWORK TO GITHUB"
echo "=========================================="
echo ""

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
    echo "✓ Git initialized"
    echo ""
fi

# Add all files
echo "Adding all files to git..."
git add .

# Show what will be committed
echo ""
echo "Files to be committed:"
git status --short | head -20
echo "... (showing first 20, there may be more)"
echo ""

# Commit
echo "Committing changes..."
git commit -m "Complete hyperparameter optimization framework with results

- Full framework: models, optimizers, utils, data loaders
- Complete results: MNIST experiments with PSO and Random Search
- Statistical analysis and publication-quality plots
- Comprehensive documentation
- Results: PSO 99.04% vs Random Search 98.68% accuracy"

echo "✓ Files committed"
echo ""

# Check if remote exists
if git remote | grep -q "origin"; then
    echo "Remote 'origin' exists. Pushing to GitHub..."
    echo ""
    echo "Current branch:"
    git branch
    echo ""
    echo "To push, run:"
    echo "  git push -u origin main"
    echo ""
    echo "Or if your branch is 'master':"
    echo "  git push -u origin master"
else
    echo "No remote 'origin' found."
    echo ""
    echo "To add your GitHub repository, run:"
    echo "  git remote add origin <YOUR_GITHUB_REPO_URL>"
    echo ""
    echo "Then push with:"
    echo "  git push -u origin main"
fi

echo ""
echo "=========================================="
echo "DONE!"
echo "=========================================="

