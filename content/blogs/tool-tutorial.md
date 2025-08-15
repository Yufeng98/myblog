---
author: "Yufeng Gu"
title: "Common Tool Tutorial"
date: 2020-06-03T08:37:58+08:00
ShowToc: true
TocOpen: false
---

This tutorial introduces three commonly used developer tools: **Git**, **Tmux** and **Vim**, as well as **Hugo**, the engine used to deploy this blog. Git handles version control and collaboration with branching and merging. Tmux keeps long-running terminal sessions alive and lets you split your screen into panes. Vim is a powerful modal editor for fast navigation and editing. Hugo is a static site generator for building and deploying websites quickly. Each section below briefly explains what the tool does, why it’s useful, and gives concise descriptions of the components you’ll use most.

## Git

Git is a distributed version control system for tracking changes in your codebase. It enables safe experimentation with branches, clear history via commits/tags, and smooth collaboration through remotes and pull/push workflows.

### Basic instructions

Quick reference for common Git commands to inspect state, view history, and interact with remotes and branches.

```shell
git [command] [--flags] [arguments]
git status [-s]
git log [--oneline] [-${num of line}]
git clone <remote_url> [localprojectname]
git remote add <name> <remote_url>
git push [-u] [<remote>] [<branch>]
git show HEAD~(parent) ~2(parent of parent) ^(first parent) ^2(second parent)
git tag <tagname> [<commit>]
git push <remote> <tagname>
git checkout -b <branch>
git branch --all
git branch -d <branch>
git merge --no-ff <branch>
git commit --amend --no-edit
```

### Create and make an initial push

Initialize a repository, create the first commit, connect a remote, and push the main branch.

```shell
mkdir machine_1
cd machine_1
git init            # initialize git in current directory
git status          # show changes of the Working Tree
# Create hello.py
git add .           # update changes in the Staging Area/Index
git commit -m "Hello World!"    # update changes in the Commit Tree
git remote add origin https://github.com/Yufeng98/git_tutorial.git
git push origin master
```

See details in [git\_tutorial](https://github.com/Yufeng98/git_tutorial) repositary.

### Version control via branch

Create feature branches, move between them, and keep history readable with concise logs.

```shell
git checkout -b cat     # create and move to a new branch named cat
# Add functionality cat to hello.py
git add .               
git commit -m "Cat"
git checkout master     # move to master branch
git checkout -b dog
# Add functionality dog to hello.py
git commit -m "dog"
git commit --amend -m "Dog"
git checkout master
git log --oneline --all --graph     # show current git log as graph
* 7d77519 (dog) Dog
| * 4a3d17c (cat) Cat
|/  
* 7ff98a6 (HEAD -> master, origin/master) Hello World
```

### Merge two branches

Integrate changes across branches and resolve conflicts when histories diverge.

```shell
git checkout -b animal
git merge cat
# animal branch is created based on master, cat branch is also based on master, so cat can be merged to animal without conflictions.
git merge dog       # fail because of confliction
vim hello.py        # deal with confliction

<<<<<<< HEAD
def cat():
    print("Cat!")

if __name__ == '__main__':
    cat()
=======
def dog();
    print("Dog!")

if __name__ == '__main__':
    dog()
>>>>>>> dog

# delete <<< === >>> and commit changes
git add .
git commit
git log --oneline --all --graph
*   e79f0bd (HEAD -> animal) Merge branch 'dog' into animal
|\  
| * 7d77519 (dog) Dog
* | 4a3d17c (cat) Cat
|/  
* 7ff98a6 (origin/master, master) Hello World

git checkout master
git merge animals
git log --oneline --all --graph
*   e79f0bd (HEAD -> master, animal) Merge branch 'dog' into animal
|\  
| * 7d77519 (dog) Dog
* | 4a3d17c (cat) Cat
|/  
* 7ff98a6 (origin/master) Hello World

git push
git log --oneline --all --graph
*   e79f0bd (HEAD -> master, origin/master, animal) Merge branch 'dog' into animal
|\  
| * 7d77519 (dog) Dog
* | 4a3d17c (cat) Cat
|/  
* 7ff98a6 Hello World
```

### Collaboration

Work across machines/remotes, synchronize history, and handle upstream changes during team development.

```shell
git clone https://github.com/Yufeng98/git_tutorial machine_2
cd machine_2
# Add functionality fish to hello.py
git add .
git commit -m "Fish"
# Add functionality duck to hello.py
git add .
git commit -m "Duck"
git reset --hard ${hash of fish}
git log --oneline --all --graph
* b9d2299 (master) Duck
* 62a9ec4 (HEAD) Fish
*   e79f0bd (origin/master, origin/HEAD) Merge branch 'dog' into animal
|\  
| * 7d77519 Dog
* | 4a3d17c Cat
|/  
* 7ff98a6 Hello World

git push origin HEAD:master
cd ../machine_1
git fetch
git log --oneline --all --graph
* 62a9ec4 (origin/master) Fish
*   e79f0bd (HEAD -> master, animal) Merge branch 'dog' into animal
|\  
| * 7d77519 (dog) Dog
* | 4a3d17c (cat) Cat
|/  
* 7ff98a6 Hello World
git merge origin/master     # git pull is similar to git fetch and merge

# add functionality tiger to hello.py and update it to remote repository
cd ../machine_2
git fetch
git log --oneline --all --graph
* f96f96d (origin/master, origin/HEAD) Tiger
| * b9d2299 (master) Duck
|/  
* 62a9ec4 (HEAD) Fish
*   e79f0bd Merge branch 'dog' into animal
|\  
| * 7d77519 Dog
* | 4a3d17c Cat
|/  
* 7ff98a6 Hello World

git checkout master
vim hello.py
<<<<<<< HEAD
def duck():
    print("Duck!")
=======
def tiger():
    print("Tiger!")
>>>>>>> f96f96dba513b1c0b6e44a7191d0ff1f3fdf33e8

if __name__ == '__main__':
    hello()
    cat()
    dog()
    fish()
<<<<<<< HEAD
    duck()
=======
    tiger()
>>>>>>> f96f96dba513b1c0b6e44a7191d0ff1f3fdf33e8

# delete <<< === >>> and commit changes
git add .
git commit
git log --oneline --all --graph
*   de3b050 (HEAD -> master) Merge branch 'master' of https://github.com/Yufeng98/git_tutorial into master
|\  
| * f96f96d (origin/master, origin/HEAD) Tiger
* | b9d2299 Duck
|/  
* 62a9ec4 Fish
*   e79f0bd Merge branch 'dog' into animal
|\  
| * 7d77519 Dog
* | 4a3d17c Cat
|/  
* 7ff98a6 Hello World

git push
cd ../machine_1
git fetch
git merge origin/master
git log --oneline --all --graph
*   de3b050 (HEAD -> master, origin/master) Merge branch 'master' of https://github.com/Yufeng98/git_tutorial into master
|\  
| * f96f96d Tiger
* | b9d2299 Duck
|/  
* 62a9ec4 Fish
*   e79f0bd (animal) Merge branch 'dog' into animal
|\  
| * 7d77519 (dog) Dog
* | 4a3d17c (cat) Cat
|/  
* 7ff98a6 Hello World
```

### Bothered by redundant commits?

Use interactive rebase to squash related commits into a cleaner history; force-push to update the remote.

```shell
git log
git rebase -i ${commit ID}
```

Replace `pick` with `squash`

Specify name of commite

```shell
git push -f
```

## Tmux

Tmux is a terminal multiplexer that lets you run multiple shells in one terminal, split views into panes, and keep sessions alive over SSH—great for remote work and long-running jobs.

### Commands

Essential command forms and management actions for creating, attaching, switching, and stopping Tmux sessions.

#### Session Management

Create, attach to, list, switch, and terminate sessions; keep workflows organized across tasks.

```shell
tmux [new -s <session-name> -m <window-name>]
tmux detach
tmux a[ttach] -t <session-name>
tmux switch -t 
tmux ls
tmux kill-session -t <session-name>
tmux kill-server # kill all sessions
```

#### Prefix Commands Ctrl+b

Tmux uses **Ctrl+b** as the default prefix; press it before the following key bindings.

#### Session

Handy key bindings to create, list, rename, and detach sessions quickly.

* `:new<Enter>`     create a new session
* `s`               list all sessions
* `$`               rename current session
* `d`               detach current session

#### Window

Manage multiple tasks per session by creating, listing, renaming, and closing windows.

* `c`               create a new window
* `w`               list all windows
* `,`               rename current window
* `&`               close current window

#### Pane

Split a window into panes for side-by-side work; close panes when done.

* `%`               splite the pane vertically
* `"`               splite the pane horizontally
* `x`               close current pane

### Configuration

Clone a ready-made configuration to get sensible defaults and productivity shortcuts.

```shell
git clone https://github.com/Yufeng98/.tmux.git
cp .tmux/.tmux.conf .
cp .tmux/.tmux.conf.local .
```

## Vim

Vim is a fast, keyboard-driven editor with modes for navigating, selecting, and editing text efficiently—ideal for coding over SSH and quick file edits.

### Normal Mode - Navigate the structure of the file

Use Normal mode to move around quickly and perform edits without leaving the keyboard.

#### Move Cursor

Essential motions to jump by character, word, line, paragraph, and page.

* `h`/`j`/`k`/`l` - Move left/up/down/right
* `H`/`M`/`L` - Move to the top/middle/bottom of the window
* `w`/`e` - Move to the beginning/end of the next word
* `b` - Move to the beggining of the previous word
* `0`/`$` - Move to the beginning/end of a line
* `(`/`)` - Move to the beggining of the next/previous line
* `{`/`}` - Move to the beginning of the next/previous paragraph
* `^E`/`^Y` - Scroll the window down/up
* `^F`/`^B` - Scroll down/up one page
* `gg`/`G` - Go to top/bottom of the file

#### Copy/Paste/Delete

Yank (copy), paste, and delete text at different granularities.

* `y` - Copy
* `p` - Paste
* `yy`/`yw`/`y$` - Copy a line/word/from the cursor to the end of the file
* `#yy` - Copy # lines
* `x` - Delete a single character
* `d`/`dd`/`dw` - Delete highlight text/a line/a word
* `D`/`d0` - Delete from the cursor to the beginning/end of the line
* `dgg`/`dG` - Delete from the cursor to the beginning/end of the file

#### Actions

Repeat and undo edits to iterate quickly and safely.

* `u`/`u#` - Undo the last/last # actions
* `.` - Repeate the last action

#### Options

Single-letter “operators/motions” to combine with objects or characters.

* `a` - all
* `i` - in
* `t` - till
* `f` - find forward
* `F` - find backard

#### Conbined Commands

Power combos that pair operators with text objects/motions for precise edits.

* `diw` - Delete inside the word
* `caw` - Change all the word
* `di)` - Delete inside the parentheses
* `dt;`  - Delete until the semicolon

### Insert Mode - Edit the file

Enter Insert mode to type and modify text directly.

* `i` - enter insert mode

### Visual Mode - Highlight portions of the file to manipulate

Select ranges of text to operate on them as a block.

* `v` - enter visual mode

### Ex Mode - Command mode

Run commands to save, quit, and perform advanced operations.

* `:q` - exit
* `:w` - save
* `:q!` - force to exit

## Hugo

Hugo is a fast static site generator that turns Markdown into a complete website. It’s great for blogs and docs, with themes, shortcodes, and near-instant local previews.

### Download Hugo

Install Hugo for your OS so you can build and serve static sites.

Follow instructions on [Hugo](https://gohugo.io/getting-started/installing).

### Verify Hugo

Confirm the installation and version.

```shell
hugo version
```

### Create a new site

Initialize a new Hugo workspace with the default directory layout.

```shell
hugo new site mypage
```

### Download a theme

Add a theme as a Git submodule so you can update it independently.

```shell
cd mypage
git init
git submodule add https://github.com/budparr/gohugo-theme-ananke.git themes/ananke
```

### Try template

Use the example site to see a working structure with sample content and assets.

We can find an example in `exampleSite`, where `static` stores images and `content` stores markdown files.

```shell
cp -r ./themes/hugo-coder/exampleSite/* ./
```

### Deploy page on local server

Run a live server with hot reload to preview your site during development.

```shell
hugo server -D
```

See it with `localhost:1313` on your browser.

### Deploy page on GitHub

Build the static files and push them to a GitHub Pages repository for hosting.

Build a repository named `${username}.github.io` and build static page configuration. Follow instructions [here](https://pages.github.com/).

```shell
hugo -D
```

The folder `public` generated with command above is what we should commit to github.

```shell
cd public
git init
git remote add origin https://github.com/${username}/${username}.github.io.git
git add .
git commit -m "Initial commit"
```

If there are existed files in repository such as `README`, conduct `pull` request first.

```shell
git pull origin master
```

Push commit. Add `-u` when pushing for the first time.

```shell
git push -u origin master
```

To support math formula, follow this [issue](https://github.com/adityatelange/hugo-PaperMod/issues/236).