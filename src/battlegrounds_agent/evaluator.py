from __future__ import annotations

from collections import Counter

from .schemas import ActionType, CandidateAction, Card, GameState, Trinket
from .synergy import score_card_synergy


class HeuristicEvaluator:
    """Fast deterministic scoring used as candidate generation and fallback."""

    def propose_actions(self, state: GameState) -> list[CandidateAction]:
        actions: list[CandidateAction] = []
        actions.extend(self.score_shop_cards(state))

        level_score, level_reason = self.score_level(state)
        if state.shop.level_cost is not None and state.player.gold >= state.shop.level_cost:
            actions.append(
                CandidateAction(
                    action_type=ActionType.LEVEL,
                    score=level_score,
                    reason=level_reason,
                    metadata={"cost": state.shop.level_cost},
                )
            )

        if state.player.gold >= state.shop.roll_cost:
            actions.append(
                CandidateAction(
                    action_type=ActionType.ROLL,
                    score=self.score_roll(state),
                    reason="商店没有明显核心牌时，刷新寻找当前阵容配合、对子或关键经济牌。",
                    metadata={"cost": state.shop.roll_cost},
                )
            )

        actions.extend(self.score_trinkets(state))
        return sorted(actions, key=lambda item: item.score, reverse=True)

    def score_shop_cards(self, state: GameState) -> list[CandidateAction]:
        board_tribes = self._board_tribes(state)
        board_tags = self._board_tags_text(state)
        actions: list[CandidateAction] = []

        for card in state.shop.cards:
            score = 10.0 + card.tier * 1.5
            reasons: list[str] = []

            tribe_overlap = board_tribes.intersection(card.tribes)
            if tribe_overlap:
                score += 7.0 + len(tribe_overlap) * 2
                reasons.append(f"与当前场面种族 {', '.join(sorted(tribe_overlap))} 有直接配合")

            if any(tag in board_tags for tag in card.tags):
                score += 4.0
                reasons.append("机制标签与现有成长、召唤、亡语或法术体系接近")

            if any(keyword in card.text for keyword in ("每当", "永久", "获得", "召唤", "战吼", "亡语", "发现")):
                score += 3.0
                reasons.append("文本包含持续收益、资源收益或触发型效果")

            synergy_score, synergy_reasons = score_card_synergy(state, card)
            if synergy_score:
                score += min(synergy_score, 24.0)
                reasons.extend(synergy_reasons[:2])

            if len(state.player.board) >= 7:
                score -= 3.0
                reasons.append("当前场面已满，购买需要考虑卖怪成本")

            actions.append(
                CandidateAction(
                    action_type=ActionType.BUY,
                    target_id=card.id,
                    target_name=card.name,
                    score=score,
                    reason="；".join(reasons) if reasons else "基础身材和酒馆等级可接受，但缺少明显配合。",
                )
            )

        return actions

    def score_trinkets(self, state: GameState) -> list[CandidateAction]:
        board_tribes = self._board_tribes(state)
        board_tags = self._board_tags_text(state)
        actions: list[CandidateAction] = []
        for trinket in state.offered_trinkets:
            score = 20.0
            reasons: list[str] = []
            overlap = board_tribes.intersection(trinket.tribes)
            if overlap:
                score += 12.0
                reasons.append(f"强化当前 {', '.join(sorted(overlap))} 体系")
            if any(tag in board_tags or tag in trinket.text for tag in trinket.tags):
                score += 6.0
                reasons.append("和当前核心机制标签一致")
            if any(word in trinket.text for word in ("每回合", "永久", "发现", "获得")):
                score += 4.0
                reasons.append("提供长期收益")
            actions.append(
                CandidateAction(
                    action_type=ActionType.PICK_TRINKET,
                    target_id=trinket.id,
                    target_name=trinket.name,
                    score=score,
                    reason="；".join(reasons) if reasons else "通用收益，和当前场面关联较弱。",
                )
            )
        return actions

    def score_level(self, state: GameState) -> tuple[float, str]:
        health_buffer = state.player.health + state.player.armor
        if health_buffer <= 15:
            return 8.0, "血量较低，优先补战力，升本风险高。"
        if state.player.tavern_tier <= 3 and health_buffer >= 28:
            return 28.0, "血量健康且处于低本阶段，升本能打开更高质量卡池。"
        if state.player.tavern_tier >= 5:
            return 12.0, "高本继续升本收益有限，除非战力已经稳定。"
        return 18.0, "血量尚可，可以在商店质量一般时考虑升本。"

    def score_roll(self, state: GameState) -> float:
        best_buy = max((self._rough_card_score(state, card) for card in state.shop.cards), default=0.0)
        return 22.0 if best_buy < 16.0 else 9.0

    def estimate_risk(self, state: GameState) -> str:
        health_buffer = state.player.health + state.player.armor
        if health_buffer <= 15:
            return "high"
        if health_buffer <= 28:
            return "medium"
        return "low"

    def infer_composition(self, state: GameState) -> str:
        counts = Counter(tribe for minion in state.player.board for tribe in minion.tribes)
        if not counts:
            return "暂未形成明确阵容，优先拿通用成长、经济牌、对子和即时战力。"
        tribe, count = counts.most_common(1)[0]
        if count >= 3:
            return f"围绕 {tribe} 建立主阵容，寻找核心成长、经济和终局组件。"
        return f"{tribe} 牌数量领先，但阵容未锁定，保留转型空间。"

    def _rough_card_score(self, state: GameState, card: Card) -> float:
        board_tribes = self._board_tribes(state)
        score = 10 + card.tier
        if board_tribes.intersection(card.tribes):
            score += 7
        return score

    def _board_tribes(self, state: GameState) -> set[str]:
        return {tribe for minion in state.player.board for tribe in minion.tribes}

    def _board_tags_text(self, state: GameState) -> str:
        board_text = [minion.text for minion in state.player.board]
        hand_tags = [tag for card in state.player.hand for tag in card.tags]
        return " ".join(board_text + hand_tags)
