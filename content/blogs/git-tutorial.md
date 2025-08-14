---
author: "Yufeng Gu"
title: "Git Tutorial"
date: 2020-08-17T08:37:58+08:00
description: "Git Tutorial"
ShowToc: true
TocOpen: false
---

### Basic instructions

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

See details in [git_tutorial](https://github.com/Yufeng98/git_tutorial) repositary.

### Version control via branch

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

```shell
git log
git rebase -i ${commit ID}
```
Replace `pick` with `squash`

Specify name of commite

```shell
git push -f
```