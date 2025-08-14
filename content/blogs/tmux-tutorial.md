---
author: "Yufeng Gu"
title: "Tmux Tutorial"
date: 2020-06-05T08:37:58+08:00
description: "Tmux Tutorial"
ShowToc: true
TocOpen: false
---

### Commands

#### Session Management

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

#### Session

* `:new<Enter>`     create a new session
* `s`               list all sessions
* `$`               rename current session
* `d`               detach current session

#### Window

* `c`               create a new window
* `w`               list all windows
* `,`               rename current window
* `&`               close current window

#### Pane

* `%`               splite the pane vertically
* `"`               splite the pane horizontally
* `x`               close current pane

### Configuration

```shell
git clone https://github.com/Yufeng98/.tmux.git
cp .tmux/.tmux.conf .
cp .tmux/.tmux.conf.local .
```