from __future__ import annotations

import json
import queue
import traceback
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from types import SimpleNamespace
from typing import Any

from PIL import Image, ImageTk

from .crop_profiles import resolve_profile
from .database import CardDatabase
from .image_recognizer import ImageRecognizer
from .live_agent import _run_once
from .llm import OpenAICompatibleClient
from .planner import BattlegroundsAgent
from .state_builder import BuildStateOptions


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CARD_DIR = r"C:\Users\dlw\Documents\Codex\2026-07-05\gen\outputs\data card"
DEFAULT_TRINKET_DIR = r"C:\Users\dlw\Documents\Codex\2026-07-05\gen\outputs\data sp"


class DesktopAssistantApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Battlegrounds Decision Agent")
        self.root.geometry("520x560")
        self.root.minsize(420, 420)

        self.events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.last_payload: dict[str, Any] | None = None
        self.database: CardDatabase | None = None
        self.recognizer: ImageRecognizer | None = None
        self.llm: OpenAICompatibleClient | None = None
        self.card_photo_refs: list[ImageTk.PhotoImage] = []
        self.settings_window: tk.Toplevel | None = None

        self.language_var = tk.StringVar(value="简体中文")
        self.phase_var = tk.StringVar(value="shop-buy-16x9")
        self.topmost_var = tk.BooleanVar(value=True)
        self.pin_top_right_var = tk.BooleanVar(value=True)
        self.hide_during_capture_var = tk.BooleanVar(value=True)
        self.use_llm_var = tk.BooleanVar(value=True)
        self.detect_game_var = tk.BooleanVar(value=True)
        self.use_template_hud_var = tk.BooleanVar(value=True)
        self.use_vision_hud_var = tk.BooleanVar(value=False)
        self.use_vision_shop_var = tk.BooleanVar(value=False)
        self.tavern_tier_var = tk.StringVar(value="4")
        self.health_var = tk.StringVar(value="30")
        self.armor_var = tk.StringVar(value="0")
        self.gold_var = tk.StringVar(value="8")
        self.turn_var = tk.StringVar(value="8")
        self.level_cost_var = tk.StringVar(value="")
        self.tribes_var = tk.StringVar(value="野兽,机械,龙,元素,海盗,鱼人,亡灵,野猪人,纳迦,恶魔")
        self.status_var = tk.StringVar(value="Ready")
        self.hud_fallback_entries: list[ttk.Entry] = []
        self.error_log_path = PROJECT_ROOT / "work" / "desktop_assistant" / "last_error.txt"

        self._build_ui()
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 0.94)
        self.root.after(50, self._apply_window_position)
        self.root.after(100, self._drain_events)

    def run(self) -> None:
        self.root.mainloop()

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)
        main.columnconfigure(0, weight=1)
        main.rowconfigure(1, weight=1)

        controls = ttk.Frame(main)
        controls.grid(row=0, column=0, sticky="ew")
        controls.columnconfigure(1, weight=1)

        output = ttk.Frame(main)
        output.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        output.rowconfigure(2, weight=1)
        output.columnconfigure(0, weight=1)

        self._build_controls(controls)
        self._build_output(output)

    def _build_controls(self, frame: ttk.Frame) -> None:
        row = 0
        buttons = ttk.Frame(frame)
        buttons.grid(row=row, column=0, columnspan=2, sticky="ew")
        buttons.columnconfigure(0, weight=1)
        buttons.columnconfigure(1, weight=1)
        buttons.columnconfigure(2, weight=0)
        ttk.Button(buttons, text="Capture & Decide", command=self.capture_and_decide).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(buttons, text="Fast Scan", command=self.fast_scan).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(buttons, text="Settings", command=self.open_settings).grid(row=0, column=2, sticky="e", padx=(4, 0))
        row += 1

        quick = ttk.Frame(frame)
        quick.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        quick.columnconfigure(0, weight=1)
        quick.columnconfigure(1, weight=1)
        quick.columnconfigure(2, weight=1)
        ttk.Button(quick, text="Refresh?", command=lambda: self.ask_question("现在应该刷新吗？")).grid(row=0, column=0, sticky="ew", padx=(0, 3))
        ttk.Button(quick, text="Level?", command=lambda: self.ask_question("现在应该升本吗？")).grid(row=0, column=1, sticky="ew", padx=3)
        ttk.Button(quick, text="Buy?", command=lambda: self.ask_question("现在应该买哪张商店牌？如果满场，应该卖谁？")).grid(row=0, column=2, sticky="ew", padx=(3, 0))
        row += 1

        ttk.Label(frame, textvariable=self.status_var, wraplength=480, foreground="#555").grid(row=row, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        self._apply_auto_hud_state()
        return

        ttk.Label(frame, text="Language").grid(row=row, column=0, sticky="w")
        ttk.Combobox(frame, textvariable=self.language_var, values=("简体中文", "English"), state="readonly", width=14).grid(row=row, column=1, sticky="ew")
        row += 1

        ttk.Label(frame, text="Phase").grid(row=row, column=0, sticky="w", pady=(8, 0))
        ttk.Combobox(
            frame,
            textvariable=self.phase_var,
            values=("shop-buy-16x9", "shop-buy", "shop-buy-right", "trinket-fullscreen", "trinket-video", "triple-discover"),
            state="readonly",
            width=16,
        ).grid(row=row, column=1, sticky="ew", pady=(8, 0))
        row += 1

        ttk.Label(frame, text="Fallback only").grid(row=row, column=0, columnspan=2, sticky="w", pady=(10, 0))
        row += 1
        for label, var in (
            ("Tavern", self.tavern_tier_var),
            ("Health", self.health_var),
            ("Armor", self.armor_var),
            ("Gold", self.gold_var),
            ("Turn", self.turn_var),
            ("Level cost", self.level_cost_var),
        ):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=(5, 0))
            entry = ttk.Entry(frame, textvariable=var, width=10)
            entry.grid(row=row, column=1, sticky="ew", pady=(5, 0))
            if label in {"Health", "Armor", "Gold"}:
                self.hud_fallback_entries.append(entry)
            row += 1

        ttk.Label(frame, text="Tribes").grid(row=row, column=0, sticky="nw", pady=(6, 0))
        ttk.Entry(frame, textvariable=self.tribes_var, width=24).grid(row=row, column=1, sticky="ew", pady=(6, 0))
        row += 1

        ttk.Checkbutton(frame, text="Use LLM", variable=self.use_llm_var).grid(row=row, column=0, columnspan=2, sticky="w", pady=(10, 0))
        row += 1
        ttk.Checkbutton(frame, text="Auto HUD local", variable=self.use_template_hud_var, command=self._apply_auto_hud_state).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        ttk.Checkbutton(frame, text="Auto HUD vision", variable=self.use_vision_hud_var).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        ttk.Checkbutton(frame, text="Vision shop exact", variable=self.use_vision_shop_var).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        ttk.Checkbutton(frame, text="Detect game area", variable=self.detect_game_var).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        ttk.Checkbutton(frame, text="Always on top", variable=self.topmost_var, command=self._apply_topmost).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        ttk.Checkbutton(frame, text="Pin top-right", variable=self.pin_top_right_var, command=self._apply_window_position).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        ttk.Checkbutton(frame, text="Hide while capturing", variable=self.hide_during_capture_var).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1

        ttk.Button(frame, text="Fast Scan", command=self.fast_scan).grid(row=row, column=0, columnspan=2, sticky="ew", pady=(14, 4))
        row += 1
        ttk.Button(frame, text="Capture & Decide", command=self.capture_and_decide).grid(row=row, column=0, columnspan=2, sticky="ew", pady=2)
        row += 1
        ttk.Button(frame, text="Refresh?", command=lambda: self.ask_question("现在应该刷新吗？")).grid(row=row, column=0, columnspan=2, sticky="ew", pady=2)
        row += 1
        ttk.Button(frame, text="Level?", command=lambda: self.ask_question("现在应该升本吗？")).grid(row=row, column=0, columnspan=2, sticky="ew", pady=2)
        row += 1
        ttk.Button(frame, text="Buy?", command=lambda: self.ask_question("现在应该买哪张商店牌？如果满场，应该卖谁？")).grid(row=row, column=0, columnspan=2, sticky="ew", pady=2)
        row += 1

        ttk.Label(frame, textvariable=self.status_var, wraplength=220, foreground="#555").grid(row=row, column=0, columnspan=2, sticky="ew", pady=(14, 0))
        self._apply_auto_hud_state()

    def open_settings(self) -> None:
        if self.settings_window is not None and self.settings_window.winfo_exists():
            self.settings_window.lift()
            return

        window = tk.Toplevel(self.root)
        self.settings_window = window
        window.title("Settings")
        window.transient(self.root)
        window.resizable(False, True)
        window.attributes("-topmost", bool(self.topmost_var.get()))

        frame = ttk.Frame(window, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        frame.columnconfigure(1, weight=1)

        row = 0
        ttk.Label(frame, text="Language").grid(row=row, column=0, sticky="w")
        ttk.Combobox(frame, textvariable=self.language_var, values=("简体中文", "English"), state="readonly", width=18).grid(row=row, column=1, sticky="ew")
        row += 1

        ttk.Label(frame, text="Phase").grid(row=row, column=0, sticky="w", pady=(8, 0))
        ttk.Combobox(
            frame,
            textvariable=self.phase_var,
            values=("shop-buy-16x9", "shop-buy", "shop-buy-right", "trinket-fullscreen", "trinket-video", "triple-discover"),
            state="readonly",
            width=20,
        ).grid(row=row, column=1, sticky="ew", pady=(8, 0))
        row += 1

        ttk.Label(frame, text="Fallback only").grid(row=row, column=0, columnspan=2, sticky="w", pady=(10, 0))
        row += 1
        for label, var in (
            ("Tavern", self.tavern_tier_var),
            ("Health", self.health_var),
            ("Armor", self.armor_var),
            ("Gold", self.gold_var),
            ("Turn", self.turn_var),
            ("Level cost", self.level_cost_var),
        ):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=(5, 0))
            entry = ttk.Entry(frame, textvariable=var, width=12)
            entry.grid(row=row, column=1, sticky="ew", pady=(5, 0))
            if label in {"Health", "Armor", "Gold"}:
                self.hud_fallback_entries.append(entry)
            row += 1

        ttk.Label(frame, text="Tribes").grid(row=row, column=0, sticky="nw", pady=(6, 0))
        ttk.Entry(frame, textvariable=self.tribes_var, width=32).grid(row=row, column=1, sticky="ew", pady=(6, 0))
        row += 1

        ttk.Checkbutton(frame, text="Use LLM", variable=self.use_llm_var).grid(row=row, column=0, columnspan=2, sticky="w", pady=(10, 0))
        row += 1
        ttk.Checkbutton(frame, text="Auto HUD local", variable=self.use_template_hud_var, command=self._apply_auto_hud_state).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        ttk.Checkbutton(frame, text="Auto HUD vision", variable=self.use_vision_hud_var).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        ttk.Checkbutton(frame, text="Vision shop exact", variable=self.use_vision_shop_var).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        ttk.Checkbutton(frame, text="Detect game area", variable=self.detect_game_var).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        ttk.Checkbutton(frame, text="Always on top", variable=self.topmost_var, command=self._apply_topmost).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        ttk.Checkbutton(frame, text="Pin top-right", variable=self.pin_top_right_var, command=self._apply_window_position).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        ttk.Checkbutton(frame, text="Hide while capturing", variable=self.hide_during_capture_var).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1

        ttk.Button(frame, text="Close", command=window.destroy).grid(row=row, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        self._apply_auto_hud_state()

    def _build_output(self, frame: ttk.Frame) -> None:
        top = ttk.Frame(frame)
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(0, weight=1)
        self.question_var = tk.StringVar()
        ttk.Entry(top, textvariable=self.question_var).grid(row=0, column=0, sticky="ew", padx=(0, 8))
        ttk.Button(top, text="Ask", command=self.ask_current_question).grid(row=0, column=1)

        self.card_frame = ttk.Frame(frame)
        self.card_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))

        self.text = tk.Text(frame, wrap=tk.WORD, height=30)
        self.text.grid(row=2, column=0, sticky="nsew", pady=(8, 0))
        scroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.text.yview)
        scroll.grid(row=2, column=1, sticky="ns", pady=(8, 0))
        self.text.configure(yscrollcommand=scroll.set)
        self._append("点击 Capture & Decide 开始识别当前局面。\n右上角固定可用 Pin top-right 控制；需要避免截图截到窗口时再勾选 Hide while capturing。\n")

    def _apply_topmost(self) -> None:
        self.root.attributes("-topmost", bool(self.topmost_var.get()))

    def _apply_window_position(self) -> None:
        if not self.pin_top_right_var.get():
            return
        self.root.update_idletasks()
        width = max(self.root.winfo_width(), 520)
        height = max(self.root.winfo_height(), 560)
        screen_width = self.root.winfo_screenwidth()
        x = max(0, screen_width - width - 16)
        y = 16
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def _apply_auto_hud_state(self) -> None:
        state = "disabled" if self.use_template_hud_var.get() else "normal"
        self.hud_fallback_entries = [entry for entry in self.hud_fallback_entries if entry.winfo_exists()]
        for entry in self.hud_fallback_entries:
            entry.configure(state=state)

    def capture_and_decide(self) -> None:
        self.status_var.set("Capturing and asking LLM...")
        self._reset_output("=== Capture & Decide ===\n正在截图和识别...\n")
        self._start_capture_worker(self.use_llm_var.get())

    def fast_scan(self) -> None:
        self.status_var.set("Fast scanning...")
        self._reset_output("=== Fast Scan ===\n正在截图和识别...\n")
        self._start_capture_worker(False)

    def _start_capture_worker(self, use_llm: bool) -> None:
        self._hide_for_capture()
        delay_ms = 300 if self.hide_during_capture_var.get() else 0
        self.root.after(
            delay_ms,
            lambda: threading.Thread(target=self._capture_worker, args=(use_llm,), daemon=True).start(),
        )

    def _hide_for_capture(self) -> None:
        if self.hide_during_capture_var.get():
            self.root.withdraw()
            self.root.update_idletasks()

    def _show_after_capture(self) -> None:
        if self.hide_during_capture_var.get():
            self.root.deiconify()
            self.root.lift()
            self._apply_topmost()
            self._apply_window_position()

    def ask_current_question(self) -> None:
        question = self.question_var.get().strip()
        if question:
            self.ask_question(question)

    def ask_question(self, question: str) -> None:
        if self.last_payload is None:
            self._reset_output("请先点击 Capture & Decide 获取当前局面。\n")
            return
        self.status_var.set("Asking LLM...")
        self._reset_output(f"=== Question ===\n{question}\n\n正在询问 LLM...\n")
        threading.Thread(target=self._ask_worker, args=(question,), daemon=True).start()

    def _capture_worker(self, use_llm: bool) -> None:
        try:
            self._ensure_runtime()
            assert self.database is not None
            assert self.recognizer is not None
            profile = resolve_profile(None, self.phase_var.get())
            agent = BattlegroundsAgent(llm_client=self.llm)
            options = BuildStateOptions(
                tavern_tier=_int(self.tavern_tier_var.get(), 4),
                health=_int(self.health_var.get(), 30),
                armor=_int(self.armor_var.get(), 0),
                gold=_int(self.gold_var.get(), 8),
                turn=_int(self.turn_var.get(), 8),
                level_cost=_optional_int(self.level_cost_var.get()),
                available_tribes=tuple(_split_csv(self.tribes_var.get())),
                min_score=0.50,
                low_confidence_score=0.60 if self.phase_var.get() == "shop-buy-16x9" else 0.68,
            )
            args = SimpleNamespace(
                screenshot=None,
                screen=True,
                window_title="炉石传说",
                detect_game_area=self.detect_game_var.get(),
                work_dir=str(PROJECT_ROOT / "work" / "desktop_assistant"),
                top_k=15,
                use_template_hud=self.use_template_hud_var.get(),
                use_vision_hud=self.use_vision_hud_var.get(),
                use_vision_shop=self.use_vision_shop_var.get(),
                use_llm=use_llm,
            )
            payload = _run_once(args, profile, self.database, self.recognizer, agent, options, self.llm)
            self.events.put(("capture", payload))
        except Exception as exc:
            self.events.put(("error", _format_exception(exc)))

    def _ask_worker(self, question: str) -> None:
        try:
            self._ensure_runtime()
            if self.llm is None:
                raise RuntimeError("LLM is not configured. Check .env.")
            prompt = {
                "language": self.language_var.get(),
                "player_question": question,
                "current_observation": _compact_payload(self.last_payload),
                "answer_rules": [
                    "Answer for the player, not for developers.",
                    "Keep it short: direct answer, next actions, brief reason.",
                    "Use only the provided current_observation.",
                    "If asking whether to level, roll, or buy, give a direct yes/no and the next 1-3 actions.",
                ],
            }
            answer = self.llm.complete(
                "你是酒馆战棋实时助手。只回答玩家当前问题，直接给结论、行动和简短理由。",
                json.dumps(prompt, ensure_ascii=False, indent=2),
            )
            self.events.put(("answer", answer.strip()))
        except Exception as exc:
            self.events.put(("error", _format_exception(exc)))

    def _ensure_runtime(self) -> None:
        if self.database is None:
            self.database = CardDatabase.load(PROJECT_ROOT / "examples" / "data" / "cards.enriched.json", PROJECT_ROOT / "examples" / "data" / "trinkets.generated.json")
        if self.recognizer is None:
            cache = PROJECT_ROOT / "work" / "recognizer_fast_index.json"
            if cache.exists():
                self.recognizer = ImageRecognizer.from_json_index(cache, use_cv=True)
            else:
                self.recognizer = ImageRecognizer.from_directories(DEFAULT_CARD_DIR, DEFAULT_TRINKET_DIR, None, use_cv=True)
                self.recognizer.write_json_index(cache)
        if self.llm is None:
            self.llm = OpenAICompatibleClient.from_env(PROJECT_ROOT / ".env")

    def _drain_events(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "capture":
                    self.last_payload = payload
                    self._show_after_capture()
                    self._render_recommended_images(payload)
                    self._render_payload(payload)
                    self.status_var.set("Decision ready")
                elif kind == "answer":
                    self._reset_output(f"=== Answer ===\n{payload}\n")
                    self.status_var.set("Answer ready")
                elif kind == "error":
                    self._show_after_capture()
                    self._log_error(str(payload))
                    self._append(f"\nERROR: {_first_error_line(str(payload))}\n完整错误已写入：{self.error_log_path}\n")
                    self.status_var.set("Error")
        except queue.Empty:
            pass
        self.root.after(100, self._drain_events)

    def _render_payload(self, payload: dict[str, Any]) -> None:
        plan = payload.get("plan", {})
        state = payload.get("state", {})
        hud = payload.get("hud", {}).get("reading")
        phase_status = payload.get("phase_status", {})
        vision_shop = payload.get("vision_shop")
        actions = plan.get("actions", [])

        buy_actions = [item for item in actions if item.get("action_type") in {"buy", "pick_trinket"}]
        level_actions = [item for item in actions if item.get("action_type") == "level"]
        roll_actions = [item for item in actions if item.get("action_type") == "roll"]

        lines = [
            "=== 建议 ===",
            f"买牌：{plan.get('buy_recommendation') or _action_targets(buy_actions) or '暂不明确'}",
            f"升本/刷新：{plan.get('level_or_roll_recommendation') or _level_roll_text(level_actions, roll_actions)}",
            f"找配合：{_synergy_text(plan) or plan.get('composition_goal') or '根据当前高置信识别牌继续补强'}",
            f"理由：{_brief_reason(plan, actions)}",
            "",
            "=== 状态 ===",
            f"识别阶段：{_phase_status_text(phase_status)}",
            _vision_shop_text(vision_shop),
            _state_summary(state),
        ]
        if hud:
            lines.append(f"HUD自动读取：金币 {hud.get('gold')}，血量 {hud.get('health')}，护甲 {hud.get('armor')}，科技 {hud.get('tavern_tier')}")
        self._append("\n".join(lines) + "\n")

    def _append(self, value: str) -> None:
        self.text.insert(tk.END, value)
        self.text.see(tk.END)

    def _reset_output(self, value: str = "") -> None:
        self.text.delete("1.0", tk.END)
        for child in self.card_frame.winfo_children():
            child.destroy()
        self.card_photo_refs = []
        if value:
            self._append(value)

    def _log_error(self, value: str) -> None:
        self.error_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.error_log_path.write_text(value, encoding="utf-8")

    def _render_recommended_images(self, payload: dict[str, Any]) -> None:
        for child in self.card_frame.winfo_children():
            child.destroy()
        self.card_photo_refs = []

        phase_status = payload.get("phase_status", {})
        if phase_status.get("phase") == "non_shop_or_uncertain":
            ttk.Label(self.card_frame, text="当前不是商店阶段：已跳过商店买牌识别").pack(anchor="w")
            return

        cards = _recommended_cards(payload)
        if not cards:
            ttk.Label(self.card_frame, text="推荐卡图：暂无明确购买目标").pack(anchor="w")
            return

        ttk.Label(self.card_frame, text="推荐购买：").pack(side=tk.LEFT, padx=(0, 8))
        for card in cards[:3]:
            path = card.get("image")
            cell = ttk.Frame(self.card_frame)
            cell.pack(side=tk.LEFT, padx=6)
            if path and Path(path).exists():
                image = Image.open(path).convert("RGB")
                image.thumbnail((82, 118), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(image)
                self.card_photo_refs.append(photo)
                ttk.Label(cell, image=photo).pack()
            ttk.Label(cell, text=str(card.get("name") or card.get("id") or ""), wraplength=95, justify=tk.CENTER).pack()


def _compact_payload(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {}
    return {"state": payload.get("state"), "plan": payload.get("plan"), "hud": payload.get("hud")}


def _format_exception(exc: Exception) -> str:
    return f"{exc}\n\n{traceback.format_exc()}"


def _first_error_line(value: str) -> str:
    for line in value.splitlines():
        line = line.strip()
        if line:
            return line[:220]
    return "未知错误"


def _action_targets(actions: list[dict[str, Any]], limit: int = 2) -> str:
    names = [str(item.get("target_name") or item.get("target_id") or item.get("action_type")) for item in actions[:limit]]
    return "，".join(name for name in names if name)


def _phase_status_text(value: dict[str, Any]) -> str:
    phase = value.get("phase")
    confidence = value.get("confidence")
    reason = value.get("reason") or ""
    if phase == "non_shop_or_uncertain":
        return f"非商店/不确定（HUD置信度 {confidence}）：{reason}"
    if phase == "shop":
        return f"商店阶段（HUD置信度 {confidence}）：{reason}"
    return "未检测"


def _vision_shop_text(value: dict[str, Any] | None) -> str:
    if not value:
        return "视觉商店：未启用"
    slots = value.get("slots") or []
    if not isinstance(slots, list):
        return "视觉商店：无有效结果"
    parts = []
    for item in slots[:7]:
        if not isinstance(item, dict):
            continue
        slot = item.get("slot")
        kind = item.get("type") or "unknown"
        name = item.get("name") or ""
        confidence = item.get("confidence")
        accepted = "OK" if item.get("accepted") else "?"
        parts.append(f"{slot}:{accepted} {kind} {name}({confidence})")
    return "视觉商店：" + ("；".join(parts) if parts else "无结果")


def _recommended_cards(payload: dict[str, Any]) -> list[dict[str, Any]]:
    plan = payload.get("plan", {})
    state = payload.get("state", {})
    actions = plan.get("actions", [])
    shop_cards = state.get("shop", {}).get("cards", [])
    by_id = {card.get("id"): card for card in shop_cards if card.get("id")}
    by_name = {card.get("name"): card for card in shop_cards if card.get("name")}

    result: list[dict[str, Any]] = []
    for action in actions:
        if action.get("action_type") not in {"buy", "pick_trinket"}:
            continue
        card = by_id.get(action.get("target_id")) or by_name.get(action.get("target_name"))
        if card and card not in result:
            result.append(card)
    return result


def _synergy_text(plan: dict[str, Any]) -> str:
    cards = plan.get("potential_synergy_cards") or []
    if not isinstance(cards, list):
        return ""
    parts = []
    for item in cards[:4]:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        reason = str(item.get("reason") or "").strip()
        if name and reason:
            parts.append(f"{name}（{reason}）")
        elif name:
            parts.append(name)
    return "；".join(parts)


def _level_roll_text(level_actions: list[dict[str, Any]], roll_actions: list[dict[str, Any]]) -> str:
    if level_actions:
        return "可以升本"
    if roll_actions:
        return "可以刷新"
    return "优先买牌/稳战力，暂不主动升本"


def _brief_reason(plan: dict[str, Any], actions: list[dict[str, Any]]) -> str:
    for action in actions:
        reason = str(action.get("reason") or "").strip()
        if reason:
            return reason[:140]
    summary = str(plan.get("summary") or "").strip()
    return summary[:140] if summary else "依据当前商店、场面和识别到的卡牌描述给出。"


def _state_summary(state: dict[str, Any]) -> str:
    player = state.get("player", {})
    shop = state.get("shop", {})
    board = player.get("board", [])
    cards = shop.get("cards", [])
    lines = [
        f"科技 {player.get('tavern_tier')} | 血量 {player.get('health')} + 护甲 {player.get('armor')} | 金币 {player.get('gold')} | 回合 {player.get('turn')}",
        "场面：" + ("；".join(f"{item.get('name')} {item.get('attack')}/{item.get('health')} {item.get('tribes')}" for item in board) or "未识别"),
        "商店：" + ("；".join(f"{item.get('name')} T{item.get('tier')} {item.get('attack')}/{item.get('health')} {item.get('tribes')}" for item in cards) or "未识别"),
    ]
    return "\n".join(lines)


def _int(value: str, default: int) -> int:
    try:
        return int(value)
    except ValueError:
        return default


def _optional_int(value: str) -> int | None:
    value = value.strip()
    if not value:
        return None
    return _int(value, 0)


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    DesktopAssistantApp().run()


if __name__ == "__main__":
    main()
