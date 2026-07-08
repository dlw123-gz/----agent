# Agent Framework Design

## Goal

根据当前回合的游戏状态，给玩家输出结构化行动建议：

- 是否升本。
- 商店卡牌买哪张、是否刷新、是否冻结。
- 当前场面如何围绕种族和卡牌文本形成阵容。
- 饰品回合四选一。
- 后续一到三回合的阵容方向。

## Boundaries

本工程只做决策层，不直接做以下事情：

- 不直接读游戏内存。
- 不直接点击游戏客户端。
- 不绑定固定截图识别模型。
- 不把卡牌数据写死在代码里。

这样可以让你先用 JSON 回放验证决策，再逐步替换输入层。

## Core Modules

`schemas.py`

定义稳定数据结构：`Card`、`Trinket`、`BoardMinion`、`GameState`、`CandidateAction`、`ActionPlan`。后续所有输入输出都围绕这些对象。

`vision_adapter.py`

把外部识别结果转换为 `GameState`。当前实现是 `JsonStateAdapter`，适合手动测试或接 OCR 后的标准化 JSON。

`database.py`

加载卡牌库和饰品库，并按当前对局可用种族、科技等级过滤。真实版本更新时只需要换数据库文件。

`evaluator.py`

确定性评分层。适合处理血量风险、升本节奏、种族重合、关键词、饰品长期收益等规则。

`prompt_builder.py`

把当前局面和候选行动转为 LLM prompt。这里负责控制模型输入格式和输出契约。

`llm.py`

LLM 抽象层。默认 `OpenAICompatibleClient` 调用 `/chat/completions`，如果你的 API 不是这个格式，只需要实现同名 `complete()` 方法。

`planner.py`

总控 agent。先跑启发式评分，再可选调用 LLM 复核，最后返回 `ActionPlan`。

## Recommended Runtime Loop

```text
capture screen
  -> recognize cards / board / tavern tier / health / gold
  -> normalize to GameState
  -> load current season database
  -> BattlegroundsAgent.plan_turn()
  -> show ActionPlan to player
  -> record state and selected action for evaluation
```

## LLM Usage Policy

建议不要让 LLM 独立决定所有事情。更稳的方式是：

1. 本地规则先给出候选行动和风险评分。
2. LLM 只负责解释复杂卡牌配合、判断阵容路线和在相近分数行动中排序。
3. 输出仍然走结构化 `ActionPlan`，避免 UI 层解析自然语言。

## Next Engineering Steps

1. 把真实卡牌库导出为 `cards.json`，补齐 `tags`。
2. 为每个版本维护 `available_tribes` 和饰品库。
3. 接入截图识别，先输出和 `examples/sample_state.json` 一样的结构。
4. 把实际对局结果回放保存下来，用于调评分权重。
5. 如果要自动化执行，再单独做 action executor，不要混进 planner。
