from __future__ import annotations

import json
from typing import Any

from .evaluator import HeuristicEvaluator
from .llm import LLMClient
from .prompt_builder import PromptBuilder
from .schemas import ActionPlan, ActionType, CandidateAction, GameState


class BattlegroundsAgent:
    def __init__(
        self,
        llm_client: LLMClient | None = None,
        evaluator: HeuristicEvaluator | None = None,
        prompt_builder: PromptBuilder | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.evaluator = evaluator or HeuristicEvaluator()
        self.prompt_builder = prompt_builder or PromptBuilder()

    def plan_turn(self, state: GameState, use_llm: bool = False) -> ActionPlan:
        candidates = self.evaluator.propose_actions(state)
        heuristic_plan = self._heuristic_plan(state, candidates)
        if not use_llm or self.llm_client is None:
            return heuristic_plan

        try:
            raw = self.llm_client.complete(
                self.prompt_builder.system_prompt,
                self.prompt_builder.build_user_prompt(state, candidates),
            )
        except Exception as exc:
            heuristic_plan.llm_reasoning = f"LLM request failed, fell back to local candidates: {exc}"
            return heuristic_plan
        try:
            return _plan_from_llm_json(raw, state)
        except Exception as exc:
            repaired = self._repair_llm_output(raw, heuristic_plan)
            if repaired is not None:
                return repaired
            heuristic_plan.llm_reasoning = f"LLM output parse failed: {exc}\n\nRaw response:\n{raw}"
            return heuristic_plan

    def _heuristic_plan(self, state: GameState, candidates: list[CandidateAction]) -> ActionPlan:
        top_actions = candidates[:5]
        composition_goal = self.evaluator.infer_composition(state)
        risk_level = self.evaluator.estimate_risk(state)
        return ActionPlan(
            summary=self._build_summary(top_actions, composition_goal),
            actions=top_actions,
            composition_goal=composition_goal,
            buy_recommendation=self._buy_recommendation(top_actions),
            level_or_roll_recommendation=self._level_or_roll_recommendation(top_actions),
            potential_synergy_cards=[],
            risk_level=risk_level,
        )

    def _build_summary(self, actions: list[CandidateAction], composition_goal: str) -> str:
        if not actions:
            return "没有明确可执行动作，保持当前场面并重新检查识图结果。"
        best = actions[0]
        target = f" {best.target_name}" if best.target_name else ""
        return f"优先 {best.action_type.value}{target}。{composition_goal}"

    def _buy_recommendation(self, actions: list[CandidateAction]) -> str | None:
        if actions and actions[0].action_type == ActionType.ROLL:
            return "暂不买，优先刷新找更强配合。"
        if actions and actions[0].action_type == ActionType.LEVEL:
            return "暂不买，优先升本。"
        if actions and actions[0].action_type == ActionType.HOLD:
            return "暂不买，先保持当前资源。"
        for action in actions:
            if action.action_type in {ActionType.BUY, ActionType.PICK_TRINKET}:
                return f"买 {action.target_name or action.target_id}"
        return "暂不买，优先刷新或等待更高质量商店。"

    def _level_or_roll_recommendation(self, actions: list[CandidateAction]) -> str | None:
        for action in actions:
            if action.action_type == ActionType.LEVEL:
                return "可以升本。"
            if action.action_type == ActionType.ROLL:
                return "可以刷新找更强配合。"
        return "暂不升本，先补战力或保留资源。"

    def _repair_llm_output(self, raw: str, fallback: ActionPlan) -> ActionPlan | None:
        if self.llm_client is None:
            return None
        fallback_json = json.dumps(_plan_to_jsonable(fallback), ensure_ascii=False, indent=2)
        prompt = (
            "Convert the previous strategy text into one valid JSON object following this schema:\n"
            '{"summary":"string","actions":[{"action_type":"buy|sell|roll|level|freeze|play|reposition|pick_trinket|hold","target_id":null,"target_name":null,"score":0,"reason":"string"}],"composition_goal":"string","risk_level":"low|medium|high","confidence":0.0,"warnings":[]}\n'
            "Use the fallback candidate JSON if the previous text is incomplete. Output JSON only.\n\n"
            f"Previous strategy text:\n{raw[:3000]}\n\nFallback candidate JSON:\n{fallback_json}"
        )
        try:
            repaired_raw = self.llm_client.complete(
                "Output only one valid JSON object. No analysis. First character must be {.",
                prompt,
            )
            return _plan_from_llm_json(repaired_raw)
        except Exception:
            return None


def _plan_from_llm_json(raw: str, state: GameState | None = None) -> ActionPlan:
    data = _parse_json_object(raw)
    name_lookup = _name_lookup(state) if state is not None else {}
    actions = [_candidate_from_json(item, name_lookup) for item in data.get("actions", []) if isinstance(item, dict)]
    risk = str(data.get("risk_level") or "medium")
    if risk not in {"low", "medium", "high"}:
        risk = "medium"
    warnings = data.get("warnings") or []
    llm_reasoning = json.dumps(
        {
            "confidence": data.get("confidence"),
            "warnings": warnings,
        },
        ensure_ascii=False,
        indent=2,
    )
    return ActionPlan(
        summary=str(data.get("summary") or "LLM 已返回决策。"),
        actions=actions,
        composition_goal=data.get("composition_goal"),
        buy_recommendation=_optional_string(data.get("buy_recommendation")),
        level_or_roll_recommendation=_optional_string(data.get("level_or_roll_recommendation")),
        potential_synergy_cards=_list_of_dicts(data.get("potential_synergy_cards")),
        risk_level=risk,  # type: ignore[arg-type]
        llm_reasoning=llm_reasoning,
    )


def _candidate_from_json(item: dict[str, Any], name_lookup: dict[str, str] | None = None) -> CandidateAction:
    action_value = str(item.get("action_type") or "hold")
    try:
        action_type = ActionType(action_value)
    except ValueError:
        action_type = ActionType.HOLD
    target_name = item.get("target_name")
    target_id = item.get("target_id")
    if not target_id and isinstance(target_name, str) and name_lookup:
        target_id = _resolve_target_id(target_name, name_lookup)
    return CandidateAction(
        action_type=action_type,
        target_id=target_id,
        target_name=target_name,
        score=_float_or_default(item.get("score"), 0.0),
        reason=str(item.get("reason") or ""),
        metadata={key: value for key, value in item.items() if key not in {"action_type", "target_id", "target_name", "score", "reason"}},
    )


def _parse_json_object(raw: str) -> dict[str, Any]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end < start:
            raise
        data = json.loads(raw[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("LLM response is not a JSON object")
    return data


def _float_or_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _name_lookup(state: GameState) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for card in state.shop.cards + state.player.hand:
        lookup[card.name] = card.id
    for minion in state.player.board:
        lookup[minion.name] = minion.card_id
    for trinket in state.offered_trinkets:
        lookup[trinket.name] = trinket.id
    return lookup


def _resolve_target_id(target_name: str, lookup: dict[str, str]) -> str | None:
    if target_name in lookup:
        return lookup[target_name]
    for name, item_id in lookup.items():
        if target_name in name or name in target_name:
            return item_id
    return None


def _plan_to_jsonable(plan: ActionPlan) -> dict[str, Any]:
    return {
        "summary": plan.summary,
        "buy_recommendation": plan.buy_recommendation,
        "level_or_roll_recommendation": plan.level_or_roll_recommendation,
        "potential_synergy_cards": plan.potential_synergy_cards,
        "actions": [
            {
                "action_type": action.action_type.value,
                "target_id": action.target_id,
                "target_name": action.target_name,
                "score": action.score,
                "reason": action.reason,
            }
            for action in plan.actions
        ],
        "composition_goal": plan.composition_goal,
        "risk_level": plan.risk_level,
        "confidence": None,
        "warnings": [],
    }
