---
author: "Yufeng Gu"
title: "Hugo Tutorial"
date: 2020-06-01T08:37:58+08:00
description: "Hugo Tutorial"
ShowToc: true
TocOpen: false
---

### Download Hugo

Follow instructions on [Hugo](https://gohugo.io/getting-started/installing).

### Verify Hugo

```shell
hugo version
```

### Create a new site

```shell
hugo new site mypage
```

### Download a theme
```shell
cd mypage
git init
git submodule add https://github.com/budparr/gohugo-theme-ananke.git themes/ananke
```

### Try template

We can find an example in `exampleSite`, where `static` stores images and `content` stores markdown files.

```shell
cp -r ./themes/hugo-coder/exampleSite/* ./
```

### Deploy page on local server

```shell
hugo server -D
```

See it with `localhost:1313` on your browser.

### Deploy page on GitHub

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
