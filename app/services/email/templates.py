"""HTML + text email templates — professional BI dashboard styling.

All templates return (subject, text_body, html_body). No external engine;
pure Python string interpolation keeps the dependency set lean and the output
deterministic for tests.
"""

from __future__ import annotations


BRAND = "Sairash BI"
COLOR_PRIMARY = "#0f172a"  # slate-900
COLOR_ACCENT = "#0ea5e9"  # sky-500
COLOR_BG = "#f8fafc"
COLOR_CARD = "#ffffff"
COLOR_TEXT = "#1e293b"
COLOR_MUTED = "#64748b"
COLOR_BORDER = "#e2e8f0"


def _wrap(title: str, preheader: str, body_html: str, cta_url: str | None = None, cta_label: str | None = None) -> str:
    cta = ""
    if cta_url and cta_label:
        cta = f"""
        <table role="presentation" cellspacing="0" cellpadding="0" style="margin:24px 0;">
          <tr><td align="center" bgcolor="{COLOR_ACCENT}" style="border-radius:8px;">
            <a href="{cta_url}" target="_blank" style="display:inline-block;padding:12px 28px;color:#ffffff;text-decoration:none;font-weight:600;font-size:14px;">{cta_label}</a>
          </td></tr>
        </table>"""
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title></head>
<body style="margin:0;padding:0;background:{COLOR_BG};font-family:Inter,Helvetica,Arial,sans-serif;color:{COLOR_TEXT};">
<div style="display:none;max-height:0;overflow:hidden;opacity:0;">{preheader}</div>
<table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="background:{COLOR_BG};padding:32px 16px;">
<tr><td align="center">
  <table role="presentation" width="600" cellspacing="0" cellpadding="0" style="max-width:600px;width:100%;background:{COLOR_CARD};border:1px solid {COLOR_BORDER};border-radius:12px;overflow:hidden;">
    <tr><td style="background:{COLOR_PRIMARY};padding:20px 28px;">
      <div style="color:#fff;font-size:18px;font-weight:700;letter-spacing:-0.02em;">{BRAND}</div>
      <div style="color:#94a3b8;font-size:12px;margin-top:2px;">Business Intelligence &amp; Decision Support</div>
    </td></tr>
    <tr><td style="padding:28px;">
      <div style="font-size:18px;font-weight:600;color:{COLOR_PRIMARY};margin-bottom:12px;">{title}</div>
      {body_html}
      {cta}
      <div style="margin-top:28px;padding-top:16px;border-top:1px solid {COLOR_BORDER};font-size:12px;color:{COLOR_MUTED};">
        This is an automated message from {BRAND}. Please do not reply directly to this email.<br>
        If you didn't expect this message, you can safely ignore it.
      </div>
    </td></tr>
  </table>
  <div style="margin-top:16px;font-size:11px;color:{COLOR_MUTED};">© {BRAND} · Asia/Kathmandu</div>
</td></tr>
</table>
</body></html>"""


def password_reset(full_name: str | None, reset_url: str) -> tuple[str, str, str]:
    from app.core.config import get_settings

    name = full_name or "there"
    ttl = getattr(get_settings(), "jwt_reset_expiry_minutes", 30)
    subject = f"{BRAND} — Reset your password"
    text = (
        f"Hi {name},\n\n"
        f"You requested a password reset for your {BRAND} account.\n"
        f"Reset link (valid for {ttl} minutes): {reset_url}\n\n"
        f"If you didn't request this, ignore this email — your password won't change.\n"
    )
    html_body = f"""
      <p style="margin:0 0 12px;color:{COLOR_TEXT};font-size:14px;line-height:1.6;">Hi {name},</p>
      <p style="margin:0 0 12px;color:{COLOR_TEXT};font-size:14px;line-height:1.6;">We received a request to reset the password for your <strong>{BRAND}</strong> account.</p>
      <p style="margin:0 0 12px;color:{COLOR_TEXT};font-size:14px;line-height:1.6;">Click the button below to set a new password. This link expires in <strong>{ttl} minutes</strong> for security.</p>
      <p style="margin:16px 0 8px;color:{COLOR_MUTED};font-size:12px;word-break:break-all;">Or copy this link:<br><a href="{reset_url}" style="color:{COLOR_ACCENT};">{reset_url}</a></p>
      <p style="margin:16px 0 0;color:{COLOR_MUTED};font-size:13px;line-height:1.6;">If you didn't request this, you can safely ignore this email — your password won't change.</p>
    """
    html = _wrap("Reset your password", f"Reset link valid for {ttl} minutes", html_body, reset_url, "Reset password →")
    return subject, text, html


def welcome(full_name: str | None, login_url: str) -> tuple[str, str, str]:
    name = full_name or "there"
    subject = f"Welcome to {BRAND}"
    text = (
        f"Hi {name},\n\n"
        f"Your {BRAND} account is ready. Sign in at {login_url} to explore your dashboards, forecasts and insights.\n\n"
        f"— The {BRAND} Team\n"
    )
    html_body = f"""
      <p style="margin:0 0 12px;color:{COLOR_TEXT};font-size:14px;line-height:1.6;">Hi {name},</p>
      <p style="margin:0 0 12px;color:{COLOR_TEXT};font-size:14px;line-height:1.6;">Your <strong>{BRAND}</strong> account is ready — you can now sign in and explore your dashboards, forecasts and decision-support insights.</p>
      <ul style="margin:12px 0;padding-left:18px;color:{COLOR_TEXT};font-size:13px;line-height:1.7;">
        <li>Track KPIs and drill down by region, product, channel and category</li>
        <li>Review forecasts, anomalies and AI-generated recommendations</li>
        <li>Schedule automated reports for your stakeholders</li>
      </ul>
    """
    html = _wrap("Welcome to Sairash BI", "Your account is ready — sign in to get started", html_body, login_url, "Open dashboard →")
    return subject, text, html


def alert_notification(rule_name: str, message: str, dashboard_url: str) -> tuple[str, str, str]:
    subject = f"[BI Alert] {rule_name}"
    text = f"{rule_name}\n\n{message}\n\nView in dashboard: {dashboard_url}\n"
    html_body = f"""
      <p style="margin:0 0 8px;color:{COLOR_MUTED};font-size:12px;letter-spacing:0.04em;text-transform:uppercase;">Alert triggered</p>
      <p style="margin:0 0 12px;color:{COLOR_PRIMARY};font-size:16px;font-weight:600;">{rule_name}</p>
      <div style="background:#fef2f2;border:1px solid #fecaca;border-radius:8px;padding:14px 16px;margin:12px 0;">
        <div style="color:#991b1b;font-size:13px;line-height:1.6;">{message}</div>
      </div>
      <p style="margin:12px 0 0;color:{COLOR_MUTED};font-size:13px;">Review the context and evidence in your dashboard.</p>
    """
    html = _wrap(rule_name, f"BI Alert: {rule_name}", html_body, dashboard_url, "View in dashboard →")
    return subject, text, html


def report_ready(title: str, period_start: str, period_end: str, fmt: str, dashboard_url: str) -> tuple[str, str, str]:
    subject = f"{BRAND} — {title} is ready"
    text = (
        f"{title}\nPeriod: {period_start} – {period_end} ({fmt.upper()})\n"
        f"Download it from your Reports page: {dashboard_url}\n"
    )
    html_body = f"""
      <p style="margin:0 0 12px;color:{COLOR_TEXT};font-size:14px;line-height:1.6;">Your scheduled report is ready for download.</p>
      <table role="presentation" cellspacing="0" cellpadding="0" style="width:100%;border:1px solid {COLOR_BORDER};border-radius:8px;overflow:hidden;margin:12px 0;">
        <tr><td style="padding:12px 16px;background:{COLOR_BG};font-size:12px;color:{COLOR_MUTED};">Report</td><td style="padding:12px 16px;font-size:13px;font-weight:600;">{title}</td></tr>
        <tr><td style="padding:12px 16px;border-top:1px solid {COLOR_BORDER};font-size:12px;color:{COLOR_MUTED};">Period</td><td style="padding:12px 16px;border-top:1px solid {COLOR_BORDER};font-size:13px;">{period_start} – {period_end}</td></tr>
        <tr><td style="padding:12px 16px;border-top:1px solid {COLOR_BORDER};font-size:12px;color:{COLOR_MUTED};">Format</td><td style="padding:12px 16px;border-top:1px solid {COLOR_BORDER};font-size:13px;">{fmt.upper()}</td></tr>
      </table>
    """
    html = _wrap(title, "Your report is ready", html_body, dashboard_url, "Open reports →")
    return subject, text, html


def verification_code(code: str, full_name: str | None = None) -> tuple[str, str, str]:
    """Fallback template for any future OTP / verification flow."""
    name = full_name or "there"
    subject = f"{BRAND} — Your verification code"
    text = f"Hi {name},\nYour verification code is: {code}\nThis code expires in 10 minutes.\n"
    html_body = f"""
      <p style="margin:0 0 12px;color:{COLOR_TEXT};font-size:14px;">Hi {name},</p>
      <p style="margin:0 0 16px;color:{COLOR_TEXT};font-size:14px;">Your verification code is:</p>
      <div style="display:inline-block;background:{COLOR_PRIMARY};color:#fff;font-size:22px;letter-spacing:0.18em;font-weight:700;padding:12px 20px;border-radius:8px;">{code}</div>
      <p style="margin:16px 0 0;color:{COLOR_MUTED};font-size:13px;">Expires in 10 minutes. If you didn't request this, ignore the email.</p>
    """
    html = _wrap("Verification code", f"Your code is {code}", html_body)
    return subject, text, html
