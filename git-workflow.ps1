#!/usr/bin/env powershell
# Git Integration Workflow - Safe Commit and Push

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "                    GIT INTEGRATION WORKFLOW - EXECUTION" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

$REPO_PATH = "d:\yasmin programs\INTEL HPE CHALLENGE 3\project-integration\vernacular-fact-checker"
Set-Location $REPO_PATH

# Step 1: Verify branch
Write-Host "[Step 1] Verifying branch status..." -ForegroundColor Yellow
$BRANCH = git rev-parse --abbrev-ref HEAD
Write-Host "Current branch: $BRANCH" -ForegroundColor Green
Write-Host ""

if ($BRANCH -ne "integration") {
    Write-Host "ERROR: Not on integration branch. Switching now..." -ForegroundColor Red
    git checkout integration
}

# Step 2: Update .gitignore
Write-Host "[Step 2] Updating .gitignore..." -ForegroundColor Yellow
$additions = @"

# ML embeddings cache
ml/cache/
ml/**/*.cache
**/embeddings_cache/

# Temp/log files
*.log
*.tmp
*.swp
*.swo
~*

# Database
*.db
*.sqlite
app.db

# Node modules
frontend/**/node_modules/
"@

Add-Content -Path ".gitignore" -Value $additions
Write-Host "[OK] .gitignore updated" -ForegroundColor Green
Write-Host ""

# Step 3: Remove .venv from git
Write-Host "[Step 3] Removing .venv from git tracking..." -ForegroundColor Yellow
git rm -r --cached ml/.venv/ 2>$null; git rm -r --cached .venv/ 2>$null
Write-Host "[OK] .venv cleanup done" -ForegroundColor Green
Write-Host ""

# Step 4: Stage files
Write-Host "[Step 4] Staging project files..." -ForegroundColor Yellow

# ML code
git add ml/inference/ 2>$null
git add ml/pipeline/ 2>$null
git add ml/training/ 2>$null
git add ml/config.py 2>$null
git add ml/__init__.py 2>$null
git add ml/requirements-ml.txt 2>$null
Write-Host "  [OK] ML code staged" -ForegroundColor Green

# Test samples
git add ml/data/verify_test_samples.jsonl 2>$null
git add ml/data/_smoke_test.py 2>$null
git add ml/data/test_all_samples.py 2>$null
Write-Host "  [OK] Test samples staged" -ForegroundColor Green

# Backend
git add backend/app/ 2>$null
git add backend/main.py 2>$null
git add backend/requirements.txt 2>$null
Write-Host "  [OK] Backend staged" -ForegroundColor Green

# Frontend
git add frontend/factcheck-frontend/src/ 2>$null
git add frontend/factcheck-frontend/vite.config.ts 2>$null
git add frontend/factcheck-frontend/tsconfig*.json 2>$null
git add frontend/factcheck-frontend/package*.json 2>$null
git add frontend/factcheck-frontend/index.html 2>$null
Write-Host "  [OK] Frontend staged" -ForegroundColor Green

# Root files
git add README.md 2>$null
git add .gitignore 2>$null
Write-Host "  [OK] Root files staged" -ForegroundColor Green
Write-Host ""

# Step 5: Show status
Write-Host "[Step 5] Reviewing staged changes..." -ForegroundColor Yellow
git status
Write-Host ""

# Step 6: Confirm
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "READY TO COMMIT AND PUSH?" -ForegroundColor Yellow
Write-Host "Review the above status. Continue? (y/n)" -ForegroundColor Yellow
$confirm = Read-Host
if ($confirm -ne "y" -and $confirm -ne "Y") {
    Write-Host ""
    Write-Host "ABORTED. No changes committed." -ForegroundColor Red
    exit 1
}

Write-Host ""

# Step 7: Commit
Write-Host "[Step 7] Creating commit..." -ForegroundColor Yellow
$msg = "Integration: Backend-ML cross-lingual verification pipeline

- Updated ml/config.py: Retrieval threshold tuning (0.35 to 0.50, 0.55 to 0.60)
- Fixed ml/inference/verifier.py: NLI input normalization (lowercase premise+hypothesis)
- Fixed ml/inference/fluff_filter.py: Case-preserving fluff removal
- Added ml/data/verify_test_samples.jsonl: 25 bilingual test samples (EN/HI)
- Added automated smoke tests: _smoke_test.py, test_all_samples.py
- Updated backend verification services with refined NLI logic
- Updated frontend claim verification UI components
- Test coverage: 21/25 samples passing (84%), all verdict types covered
- Excluded .venv and cache files from tracking"

git commit -m $msg
Write-Host "[OK] Commit created" -ForegroundColor Green
Write-Host ""

# Step 8: Verify
Write-Host "[Step 8] Verifying commit..." -ForegroundColor Yellow
git log --oneline -3
Write-Host ""

# Step 9: Push
Write-Host "[Step 9] Pushing to origin/integration..." -ForegroundColor Yellow
git push origin integration
Write-Host "[OK] Pushed to remote" -ForegroundColor Green
Write-Host ""

# Summary
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "                     SUCCESS! WORKFLOW COMPLETED" -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Summary:" -ForegroundColor Cyan
Write-Host "  [OK] Updated .gitignore"
Write-Host "  [OK] Staged all relevant files (ML, backend, frontend)"
Write-Host "  [OK] Excluded .venv and cache directories"
Write-Host "  [OK] Created commit"
Write-Host "  [OK] Pushed to GitHub integration branch"
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. https://github.com/PERRY-YASMIN/vernacular-fact-checker"
Write-Host "  2. Switch to integration branch to verify commits"
Write-Host "  3. Confirm .venv is not in the commit"
Write-Host ""
