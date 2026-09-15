# Git 协作规范

整理自小组 PDF《Python小组 Git 开发协作规范》。仓库托管 GitHub（课程要求提交链接）。Gitee 仅作国内镜像时再加，日常只维护 GitHub。

## 原则

- `main` 是保护分支，**禁止**直接提交、修改、推送。
- 功能都从最新 `main` 拉 `feat-xxx`。
- 合入 `main` 必须走 Pull Request，组长审核。
- 分支名：`feat-模块名`。本项目固定：
  - `feat-backend` 崔亚鹏
  - `feat-crawler` 滕佳杰（数据与图谱）
  - `feat-frontend` 梁先楷
  - `feat-ai` 周璇
  - 跨模块或文档可用 `feat-docs`、`fix-xxx`

一个人长期一条功能分支可以，但合并前必须同步 `main`。不要四个人共用一个 `dev`。

## 首次

```bash
git clone <组长提供的仓库地址>
cd Mars-KG-Vis
```

## 日常

```bash
git checkout main
git pull
git checkout -b feat-xxx          # 已有分支则 git checkout feat-xxx
# ... 改代码 ...
git add .
git commit -m "feat: 完成滑坡列表接口"
git push origin feat-xxx
```

然后在 GitHub：Compare & pull request → base `main`，compare `feat-xxx`。

PR 标题与 commit 一致，例如 `feat: 完成 AI 智能体基础调用`。描述写：做了什么、怎么测的、是否改库表/依赖。

## 提交说明

| 前缀 | 用途 |
| --- | --- |
| `feat:` | 新功能 |
| `fix:` | 修 bug |
| `docs:` | 文档 |
| `merge:` | 解决与 main 的冲突 |

禁止：「更新代码」「修改」「改了一点」。

## 冲突

在**自己的功能分支**解决，不要在 `main` 上解。

```bash
git checkout main
git pull
git checkout feat-xxx
git merge main
# 编辑冲突文件，去掉 <<<<<<< ======= >>>>>>>
git add .
git commit -m "merge: 解决与 main 的代码冲突"
git push origin feat-xxx
```

禁止不解决冲突就强制合并，禁止绕过 PR 合入 `main`。

## 速查

```bash
git status
git branch
git log --oneline
```

## 禁止

- 未 pull 最新 `main` 就开分支
- 提交 `.env`、数据 dump、`node_modules`、`__pycache__`、Chroma 目录
- 用 `git push --force` 推 `main`
- 把别人正在进行的功能分支强行改掉

## 组长审核清单（短）

- 能本地跑起来或说明为何这次只含脚本
- 没有密钥
- 接口变更已改 `docs/03-里程碑与接口约定.md`
- 前端字段与接口文档一致
