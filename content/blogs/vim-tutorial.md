---
author: "Yufeng Gu"
title: "Vim Tutorial"
date: 2020-06-03T08:37:58+08:00
description: "Vim Tutorial"
ShowToc: true
TocOpen: false
---

### Normal Mode - Navigate the structure of the file

#### Move Cursor
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
* `y` - Copy
* `p` - Paste
* `yy`/`yw`/`y$` - Copy a line/word/from the cursor to the end of the file
* `#yy` - Copy # lines
* `x` - Delete a single character
* `d`/`dd`/`dw` - Delete highlight text/a line/a word
* `D`/`d0` - Delete from the cursor to the beginning/end of the line
* `dgg`/`dG` - Delete from the cursor to the beginning/end of the file

#### Actions
* `u`/`u#` - Undo the last/last # actions
* `.` - Repeate the last action

#### Options
* `a` - all
* `i` - in
* `t` - till
* `f` - find forward
* `F` - find backard

#### Conbined Commands
* `diw` - Delete inside the word
* `caw` - Change all the word
* `di)` - Delete inside the parentheses
* `dt;`  - Delete until the semicolon

### Insert Mode - Edit the file
* `i` - enter insert mode

### Visual Mode - Highlight portions of the file to manipulate
* `v` - enter visual mode

### Ex Mode - Command mode
* `:q` - exit
* `:w` - save
* `:q!` - force to exit