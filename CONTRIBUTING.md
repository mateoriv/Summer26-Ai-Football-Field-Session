# Contributing Guide — Summer26 AI Football Field Session

Repository:
https://github.com/mateoriv/Summer26-Ai-Football-Field-Session

---

# IMPORTANT TEAM RULE

Nobody codes directly on `main`.

Every teammate works on their OWN branch.

---

# INITIAL SETUP (EVERY TEAM MEMBER)

## 1. Go to your projects folder

```bash
cd ~/Documents/Mines/CSCI_370
```

---

## 2. Clone the repository

```bash
git clone https://github.com/mateoriv/Summer26-Ai-Football-Field-Session.git
```

---

## 3. Enter the project folder

```bash
cd Summer26-Ai-Football-Field-Session
```

---

## 4. Install Git LFS

VERY IMPORTANT for this project because it contains ML models and large files.

```bash
git lfs install
git lfs pull
```

---

# CREATING YOUR OWN BRANCH

Each person creates their own branch.

## Example branch names

```bash
git checkout -b mateo
```

```bash
git checkout -b will
```

```bash
git checkout -b caden
```

```bash
git checkout -b toan
```

This command BOTH:

* creates the branch
* switches you onto it

---

# VERIFY YOUR CURRENT BRANCH

```bash
git branch
```

Example output:

```bash
main
* mateo-ui-work
```

The `*` means your current branch.

---

# NORMAL WORKFLOW

## 1. Edit code normally

Work on files as usual.

---

## 2. Save your work

```bash
git add .
git commit -m "Describe what you changed"
```

Example:

```bash
git commit -m "Added YOLO player detection improvements"
```

---

## 3. Push your branch

FIRST TIME ONLY:

```bash
git push -u origin mateo-ui-work
```

After the first push:

```bash
git push
```

---

# OPENING A PULL REQUEST

Go to:

https://github.com/mateoriv/Summer26-Ai-Football-Field-Session/pulls

GitHub will usually show:

> Compare & pull request

Click it.

Then:

* base branch = `main`
* compare branch = your branch

Create the Pull Request.

---

# MERGING

After review/testing:

* merge into `main`
* optionally delete merged branch

---

# DAILY TEAM WORKFLOW

## EVERY DAY BEFORE STARTING WORK

### 1. Switch to main

```bash
git checkout main
```

### 2. Pull newest updates

```bash
git pull
```

### 3. Switch back to your branch

```bash
git checkout mateo
```

Replace with YOUR branch name.

### 4. Update your branch with newest main changes

```bash
git merge main
```

Now your branch contains the newest updates.

---

# IMPORTANT COMMANDS

## Show all branches

```bash
git branch
```

## Switch branches

```bash
git checkout branch-name
```

Example:

```bash
git checkout main
```

## Create and switch to a new branch

```bash
git checkout -b new-branch-name
```

## Check modified files

```bash
git status
```

## Pull newest changes

```bash
git pull
```

## Push changes

```bash
git push
```

---

# IMPORTANT TEAM RULES

## NEVER:

* push directly to `main`
* force push to `main`
* delete other people’s branches
* work on the same files simultaneously without communication

## ALWAYS:

* pull before starting work
* use your own branch
* commit often
* push regularly
* create pull requests
* communicate merges with the team

---

# EXAMPLE FULL WORK SESSION

## Start work

```bash
git checkout main
git pull
git checkout mateo
git merge main
```

## Make changes

(edit files)

## Save work

```bash
git add .
git commit -m "Improved field line detection"
```

## Push changes

```bash
git push
```

## Open Pull Request on GitHub

Merge into `main` after review/testing.
