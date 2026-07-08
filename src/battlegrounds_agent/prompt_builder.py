from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from .composition_knowledge import load_composition_guides, relevant_compositions
from .schemas import CandidateAction, GameState
from .synergy import explain_state_synergies


DECISION_SKILL = """
你是炉石传说酒馆战棋商店阶段决策 agent。

你的输入已经由本地视觉工具处理过：
1. 视觉工具只负责识别画面中的卡牌/饰品图片 ID。
2. 系统会用 ID 查询本地 JSON 数据库，补全名称、等级、攻击、生命、种族、描述和机制标签。
3. 你不要再猜图片内容，只基于输入中的结构化 game_state 做决策。

当前只处理商店购买阶段、三连发现阶段、饰品选择阶段和阵容推荐；暂不处理战斗阶段模拟。

你理解这些机制：
- 三连：三张同名随从合成金色随从，并发现高一级酒馆随从。是否保对子/找三连要权衡当前战力、格子、金币和发现收益。
- 发现：从三个选项里选一个。优先级取决于当前血量、战力缺口、阵容核心、经济和转型空间。
- 战吼：打出时触发。布莱恩或战吼翻倍效果会显著提高战吼经济牌、发现牌、增益牌价值。
- 亡语：死亡时触发。瑞文、复生、召唤物和亡语放大效果会提高亡语体系价值。
- 经济：额外金币、铸币、出售收益、免费刷新、手牌资源可以支持升本、找核心或刷三连。
- 即战力：低血量或连败时优先身材、圣盾、嘲讽、复生、烈毒/剧毒、亡语召唤和马上能提升场面的牌。
- 阵容路线：不要过早锁死阵容；当已有核心、多个同种族组件或明确饰品/英雄技能支持时再收敛。
""".strip()


REWARD_FUNCTION = {
    "survival_and_tempo": {
        "weight": 30,
        "description": "当前血量低、场面弱或下一战压力大时，提高即战力、补盾墙、补关键身材和避免贪经济的权重。",
    },
    "economy_and_efficiency": {
        "weight": 20,
        "description": "金币利用率、买卖收益、免费刷新、铸币、保留后续操作空间；避免无意义剩钱。",
    },
    "synergy_and_scaling": {
        "weight": 25,
        "description": "已有场面与商店牌的种族、机制、成长、战吼/亡语/法术/磁力/召唤配合。",
    },
    "triples_and_discover": {
        "weight": 10,
        "description": "对子、三连、发现高本核心和发现时机；不要为了低价值三连损失过多战力。",
    },
    "leveling_and_direction": {
        "weight": 10,
        "description": "升本收益、当前酒馆等级、血量缓冲、商店质量、是否需要找高本核心。",
    },
    "uncertainty_penalty": {
        "weight": -20,
        "description": "如果识图置信度低、卡牌明显不合法或输入自相矛盾，要降低结论确定性，并建议先重新识图/提高阈值。",
    },
}


OUTPUT_CONTRACT = {
    "summary": "一句话给出本回合主计划。",
    "buy_recommendation": "建议购买的商店卡牌名称；如果不买，写不买和原因，必须简短。",
    "level_or_roll_recommendation": "直接写升本/刷新/不升本不刷新，并给一句短理由。",
    "potential_synergy_cards": [
        {
            "name": "未来要找的配合卡或核心组件名，可以来自阵容知识库或卡牌机制理解",
            "reason": "为什么它和当前场面/商店/路线配合",
        }
    ],
    "actions": [
        {
            "action_type": "buy | sell | roll | level | freeze | play | reposition | pick_trinket | hold",
            "target_id": "目标卡牌/饰品 ID，没有则为 null",
            "target_name": "目标名称，没有则为 null",
            "score": "0-100 的决策置信/奖励分",
            "reason": "为什么这样做，必须结合当前场面和卡牌描述",
        }
    ],
    "composition_goal": "当前推荐阵容方向和未来 2-3 回合找牌重点。",
    "risk_level": "low | medium | high",
    "confidence": "0-1，输入识别不可靠时必须降低",
    "warnings": ["识图、血量、金币、酒馆等级或候选动作存在的问题"],
}


class PromptBuilder:
    system_prompt = (
        "CRITICAL OUTPUT RULES:\n"
        "Output only one valid JSON object. The first character must be { and the last character must be }.\n"
        "Do not output analysis, reasoning traces, Markdown, code fences, or any text outside JSON.\n"
        "Put concise reasoning only inside each action.reason string.\n\n"
        + DECISION_SKILL
        + "\n\n你必须只输出一个合法 JSON 对象，不要 Markdown，不要代码块，不要先解释。"
        + '\n输出示例：{"summary":"先买核心牌并保留对子。","buy_recommendation":"买卡名。","level_or_roll_recommendation":"不升本，先补战力。","potential_synergy_cards":[{"name":"核心卡名","reason":"和当前机制配合"}],"actions":[{"action_type":"buy","target_id":"card_id","target_name":"卡名","score":80,"reason":"理由"}],"composition_goal":"阵容方向","risk_level":"low","confidence":0.8,"warnings":[]}'
        + "\n如果本地启发式候选动作明显不合理，可以否决它。"
    )

    def __init__(self, composition_guides_path: str | None = None) -> None:
        self.composition_guides = load_composition_guides(composition_guides_path)

    def build_user_prompt(self, state: GameState, candidates: list[CandidateAction]) -> str:
        matched_compositions = relevant_compositions(state, self.composition_guides, limit=5)
        data = {
            "task": "根据当前结构化状态，为本回合给出可执行决策。候选动作只是参考，最终以你的判断为准。",
            "game_state": _compact_state(state),
            "shop_rules": {
                "buy_minion_cost": 3,
                "sell_minion_refund": 1,
                "roll_cost": state.shop.roll_cost,
                "level_cost": state.shop.level_cost,
                "current_gold": state.player.gold,
                "current_tavern_tier": state.player.tavern_tier,
                "health_plus_armor": state.player.health + state.player.armor,
            },
            "reward_function": REWARD_FUNCTION,
            "detected_synergies": [_compact_synergy(item) for item in explain_state_synergies(state)[:10]],
            "matched_composition_guides_from_image_database": [item.to_prompt_dict() for item in matched_compositions],
            "candidate_actions_ranked_by_local_heuristic": [_compact_action(item) for item in candidates[:8]],
            "output_contract": OUTPUT_CONTRACT,
            "decision_notes": [
                "默认把 game_state 当作当前可用观测，并基于其中的卡牌描述、血量、酒馆等级、金币和场面做决策。",
                "买一个商店随从固定花费 3 金币；如果当前金币不足 3，不要建议购买随从。",
                "必须明确判断本回合是买随从、升本、刷新、冻结还是暂缓，并说明找哪些潜在配合卡。",
                "最终给玩家看的内容要短：买牌建议、升本/刷新建议、找什么配合卡、短理由。",
                "matched_composition_guides_from_image_database 来自 data team 阵容图片的视觉解析；它是阵容路线参考，不是必须照抄。",
                "notes 中的低置信度或异常等级只作为可靠性提示，不要因此拒绝决策；可以在 warnings 中简短说明。",
                "满场时购买动作必须考虑卖哪个随从或是否只是冻结/刷新。",
                "不要只说阵容名，必须给出动作顺序。",
            ],
        }
        return json.dumps(data, ensure_ascii=False, indent=2)


def _compact_state(state: GameState) -> dict[str, Any]:
    return {
        "player": {
            "tavern_tier": state.player.tavern_tier,
            "health": state.player.health,
            "armor": state.player.armor,
            "gold": state.player.gold,
            "turn": state.player.turn,
            "board": [
                {
                    "id": minion.card_id,
                    "name": minion.name,
                    "attack": minion.attack,
                    "health": minion.health,
                    "position": minion.position,
                    "tribes": minion.tribes,
                    "text": minion.text,
                    "golden": minion.golden,
                }
                for minion in state.player.board
            ],
            "hand": [_compact_card(card) for card in state.player.hand],
        },
        "shop": {
            "cards": [_compact_card(card) for card in state.shop.cards],
            "frozen": state.shop.frozen,
            "roll_cost": state.shop.roll_cost,
            "level_cost": state.shop.level_cost,
        },
        "available_tribes": state.available_tribes,
        "offered_trinkets": [
            {
                "id": item.id,
                "name": item.name,
                "text": item.text,
                "tier": item.tier,
                "tribes": item.tribes,
                "tags": item.tags,
            }
            for item in state.offered_trinkets
        ],
        "notes": state.notes,
    }


def _compact_card(card) -> dict[str, Any]:
    return {
        "id": card.id,
        "name": card.name,
        "tier": card.tier,
        "attack": card.attack,
        "health": card.health,
        "tribes": card.tribes,
        "text": card.text,
        "tags": card.tags,
    }


def _compact_action(action: CandidateAction) -> dict[str, Any]:
    return {
        "action_type": action.action_type.value,
        "target_id": action.target_id,
        "target_name": action.target_name,
        "score": action.score,
        "reason": action.reason,
    }


def _compact_synergy(item) -> dict[str, Any]:
    data = asdict(item)
    return {
        "source": data["source_name"],
        "target": data["target_name"],
        "score": data["score"],
        "reason": data["reason"],
        "tags": data["tags"],
    }
