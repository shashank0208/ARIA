import json
from datetime import datetime
from aria.agents.base import BaseAgent, AgentConfig, AgentResult


class AlertingAgent(BaseAgent):
    """
    Aggregates results from all upstream agents and fires alerts.
    Supported channels: console (default), json file, slack webhook, email.
    LLM is NOT used here — alerts must be factual, deterministic, and fast.
    """

    def run(self, adapter, context: dict) -> AgentResult:
        drift_result  = context.get("drift_result")
        perf_result   = context.get("performance_result")
        expl_result   = context.get("explainability_result")

        channels: list[str] = self.config.extra.get("channels", ["console"])

        # ── Step 1: Collect alerts from upstream agent results ─────────────────
        alerts: list[dict] = []

        for name, result in [
            ("drift",          drift_result),
            ("performance",    perf_result),
            ("explainability", expl_result),
        ]:
            if result is None:
                continue
            if result.status in ("warning", "critical", "error"):
                alerts.append({
                    "level": result.status,
                    "agent": name,
                    "summary": self._summarise(result),
                    "timestamp": result.timestamp,
                })

        overall_status = "ok"
        if any(a["level"] == "critical" for a in alerts):
            overall_status = "critical"
        elif any(a["level"] == "error" for a in alerts):
            overall_status = "error"
        elif any(a["level"] == "warning" for a in alerts):
            overall_status = "warning"

        # ── Step 2: Fire to each configured channel ────────────────────────────
        fired_to: list[str] = []

        for channel in channels:
            try:
                if channel == "console":
                    self._fire_console(alerts, overall_status)
                    fired_to.append("console")

                elif channel == "json":
                    path = self.config.extra.get("json_path", "./aria_alerts.json")
                    self._fire_json(alerts, overall_status, path)
                    fired_to.append(f"json:{path}")

                elif channel == "slack":
                    webhook = self.config.extra.get("slack_webhook_url")
                    if webhook:
                        self._fire_slack(alerts, overall_status, webhook)
                        fired_to.append("slack")

                elif channel == "email":
                    # Email integration placeholder — extend via config
                    fired_to.append("email:not_configured")

            except Exception as e:
                fired_to.append(f"{channel}:FAILED({e})")

        explanation = (
            f"No alerts — all systems nominal."
            if not alerts else
            f"{len(alerts)} alert(s) fired: " +
            ", ".join(f"{a['agent']}={a['level']}" for a in alerts)
        )

        return AgentResult(
            agent_name="AlertingAgent",
            status=overall_status,
            metrics={
                "n_alerts": len(alerts),
                "alerts": alerts,
                "fired_to": fired_to,
            },
            explanation=explanation,
        )

    def _summarise(self, result) -> str:
        """Short single-line summary from an AgentResult."""
        base = f"{result.agent_name}: {result.status.upper()}"
        if result.explanation:
            # Take first sentence only
            first_sentence = result.explanation.split(".")[0]
            return f"{base} — {first_sentence}"
        return base

    def _fire_console(self, alerts: list[dict], overall: str) -> None:
        SEP = "─" * 60
        STATUS_ICON = {"ok": "✓", "warning": "⚠", "critical": "✖", "error": "✖"}
        print(f"\n{SEP}")
        print(f"  ARIA ALERT — {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"  Overall Status: {STATUS_ICON.get(overall, '?')} {overall.upper()}")
        print(SEP)
        if not alerts:
            print("  All systems nominal. No issues detected.")
        for a in alerts:
            icon = STATUS_ICON.get(a["level"], "?")
            print(f"  {icon} [{a['level'].upper()}] {a['summary']}")
        print(f"{SEP}\n")

    def _fire_json(self, alerts: list[dict], overall: str, path: str) -> None:
        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": overall,
            "alerts": alerts,
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    def _fire_slack(self, alerts: list[dict], overall: str, webhook_url: str) -> None:
        import urllib.request
        EMOJI = {"ok": ":white_check_mark:", "warning": ":warning:", "critical": ":red_circle:", "error": ":x:"}
        lines = [f"{EMOJI.get(overall, '')} *ARIA Alert* — Overall: {overall.upper()}"]
        for a in alerts:
            lines.append(f"• {EMOJI.get(a['level'], '')} `{a['agent']}` — {a['summary']}")
        payload = json.dumps({"text": "\n".join(lines)}).encode()
        req = urllib.request.Request(webhook_url, data=payload, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=5)
