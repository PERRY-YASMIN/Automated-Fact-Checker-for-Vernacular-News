================================================================================
               GIT INTEGRATION WORKFLOW - STEP-BY-STEP COMMANDS
================================================================================
Current Status: On branch 'integration', many uncommitted changes
Goal: Safely commit and push relevant project files to origin/integration

================================================================================
STEP 1: Update .gitignore (if not already complete)
================================================================================

The .gitignore already has most exclusions, but we should add ml/cache:

Command:
--------
cd "d:\yasmin programs\INTEL HPE CHALLENGE 3\project-integration\vernacular-fact-checker"
cat >> .gitignore << EOF

# ML embeddings cache
ml/cache/
ml/**/*.cache
**/embeddings_cache/

# Common temp/log files
*.log
*.tmp
*.swp
*.swo
~*

# Database files (local SQLite)
*.db
*.sqlite
app.db

# Node modules
frontend/**/node_modules/

EOF

What this does: Appends additional exclusions for ML caches, logs, DB files, and node_modules


================================================================================
STEP 2: Remove .venv from git tracking (if already committed)
================================================================================

Command:
--------
git rm -r --cached ml/.venv/
git commit -m "chore: Remove .venv from tracking (already in .gitignore)"

What this does: Untrracks the .venv directory while keeping it on disk


================================================================================
STEP 3: Stage all relevant project files (exclude .venv and cache)
================================================================================

Commands (to run individually or together):
-------------------------------------------

# Stage ML pipeline and inference code
git add ml/inference/
git add ml/pipeline/
git add ml/training/
git add ml/config.py
git add ml/__init__.py
git add ml/requirements-ml.txt

# Stage test samples and validation
git add ml/data/verify_test_samples.jsonl
git add ml/data/_smoke_test.py
git add ml/data/test_all_samples.py
git add ml/data/diagnose_sample.py
git add ml/data/test_sample2_nli.py

# Stage backend code
git add backend/app/
git add backend/main.py
git add backend/requirements.txt

# Stage frontend code and config
git add frontend/factcheck-frontend/src/
git add frontend/factcheck-frontend/vite.config.ts
git add frontend/factcheck-frontend/tsconfig*.json
git add frontend/factcheck-frontend/package.json
git add frontend/factcheck-frontend/package-lock.json
git add frontend/factcheck-frontend/index.html
git add frontend/factcheck-frontend/eslint.config.js

# Stage project root files
git add README.md
git add .gitignore

# Verify staging (review before commit)
git status

What this does: Stages all code, config, and test files without venv or caches


================================================================================
STEP 4: Unstage any unwanted files (if they appear in git status)
================================================================================

If git status shows any unwanted files, use:
-------
git reset -- <path-to-file>

Examples:
---------
git reset -- ml/.venv/
git reset -- ml/cache/
git reset -- backend/app.db


================================================================================
STEP 5: Verify staging is correct
================================================================================

Command:
--------
git status

Expected output:
-----------------
Changes to be committed:
  modified:   backend/app/...
  modified:   ml/inference/...
  new file:   ml/data/verify_test_samples.jsonl
  ... (other code files)

NOT expected (these should NOT appear):
  - .venv files
  - __pycache__ directories
  - ml/cache files
  - *.log, *.db files


================================================================================
STEP 6: Commit with descriptive message
================================================================================

Command:
--------
git commit -m "Integration: Backend–ML cross-lingual verification pipeline

- Updated ml/config.py: Retrieval threshold tuning (0.35→0.50, 0.55→0.60)
- Fixed ml/inference/verifier.py: Normalize NLI inputs (lowercase both premise+hypothesis)
- Fixed ml/inference/fluff_filter.py: Case-preserving fluff removal
- Added ml/data/verify_test_samples.jsonl: 25 bilingual test samples (EN/HI)
- Added automated smoke tests: _smoke_test.py, test_all_samples.py
- Updated backend services: verification_service.py with refined NLI logic
- Updated frontend: Claim verification UI components
- Test coverage: 21/25 samples passing (84%), all verdict types covered
- Core fixes: NLI case sensitivity, retrieval noise filtering
"

What this does: Creates a detailed commit capturing all changes


================================================================================
STEP 7: Verify local commit
================================================================================

Command:
--------
git log --oneline -5

Expected: Your new commit should appear at the top


================================================================================
STEP 8: Push to remote (integration branch)
================================================================================

Command:
--------
git push origin integration

Expected output:
-----------------
Enumerating objects: X done.
Counting objects: 100% (X/X) done.
Delta compression using up to Y threads.
Compressing objects: 100% (X/X) done.
Writing objects: 100% (X/X) done.
remote: Create a pull request for 'integration' on GitHub by visiting:
remote: https://github.com/PERRY-YASMIN/vernacular-fact-checker/pull/new/integration
...
[integration <commit-hash>] Integration: Backend–ML cross-lingual verification pipeline


================================================================================
OPTIONAL: After push - verify on GitHub
================================================================================

1. Go to: https://github.com/PERRY-YASMIN/vernacular-fact-checker
2. Switch to 'integration' branch
3. Verify all new commits appear
4. Check that .venv/ and caches are NOT in the file list


================================================================================
                          QUICK REFERENCE (All commands)
================================================================================

# Run all staging commands at once (safer than one by one):
cd "d:\yasmin programs\INTEL HPE CHALLENGE 3\project-integration\vernacular-fact-checker"

# Update .gitignore
cat >> .gitignore << EOF

# ML embeddings cache
ml/cache/
ml/**/*.cache
**/embeddings_cache/
*.log
*.tmp
*.swp
*.swo
~*
*.db
*.sqlite
app.db
frontend/**/node_modules/
EOF

# Remove .venv from git tracking
git rm -r --cached ml/.venv/ 2>$null

# Stage files
git add ml/inference/ ml/pipeline/ ml/training/ ml/config.py ml/__init__.py ml/requirements-ml.txt
git add ml/data/verify_test_samples.jsonl ml/data/_smoke_test.py ml/data/test_all_samples.py ml/data/diagnose_sample.py
git add backend/app/ backend/main.py backend/requirements.txt
git add frontend/factcheck-frontend/src/ frontend/factcheck-frontend/*.ts frontend/factcheck-frontend/*.json frontend/factcheck-frontend/*.html frontend/factcheck-frontend/*.js
git add README.md .gitignore

# Review
git status

# Commit
git commit -m "Integration: Backend–ML cross-lingual verification pipeline

- Updated ml/config.py: Retrieval threshold tuning
- Fixed ml/inference/verifier.py: NLI input normalization  
- Fixed ml/inference/fluff_filter.py: Case-preserving fluff removal
- Added ml/data/verify_test_samples.jsonl: 25 bilingual test samples
- Added automated smoke tests (_smoke_test.py, test_all_samples.py)
- Updated backend verification services
- Updated frontend verification components
- Test coverage: 21/25 samples passing (84%)
"

# Verify commit
git log --oneline -3

# Push
git push origin integration


================================================================================
                              SAFETY CHECKLIST
================================================================================

Before pushing, verify:

[ ] .venv/ directory is NOT in git staging (git status should not show it)
[ ] __pycache__/ is NOT in staging
[ ] ml/cache/ is NOT in staging
[ ] No *.db or app.db files in staging
[ ] All .py files from ml/inference/ and ml/pipeline/ ARE in staging
[ ] All backend/app/*.py files ARE in staging
[ ] Frontend src/ and config files ARE in staging
[ ] README.md and .gitignore ARE in staging

After pushing, verify on GitHub:

[ ] Branch 'integration' has new commits
[ ] File list does not show .venv/ or cache directories
[ ] All Python source files are present
[ ] Commit message is descriptive and detailed


================================================================================
