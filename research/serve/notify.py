# -*- coding: utf-8 -*-
"""
邮件通知模块 (复用项目 .env 中的 SMTP 配置)
SMTP_USER / SMTP_PASSWORD / RECEIVER_EMAIL / SMTP_SERVER / SMTP_PORT
"""
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
except Exception:
    pass


def send_email_html(subject, html_body):
    sender = os.getenv("SMTP_USER")
    receiver = os.getenv("RECEIVER_EMAIL")
    password = os.getenv("SMTP_PASSWORD")
    server = os.getenv("SMTP_SERVER", "smtp.qq.com")
    port = int(os.getenv("SMTP_PORT", "465"))

    if not all([sender, receiver, password]):
        print("[notify] SMTP 配置缺失 (SMTP_USER/RECEIVER_EMAIL/SMTP_PASSWORD 需在 .env)")
        return False

    msg = MIMEMultipart("alternative")
    msg["From"] = sender
    msg["To"] = receiver
    msg["Subject"] = subject
    msg.attach(MIMEText(html_body, "html", "utf-8"))
    try:
        s = smtplib.SMTP_SSL(server, port, timeout=30)
        s.login(sender, password)
        s.sendmail(sender, receiver, msg.as_string())
        s.quit()
        print(f"[notify] 邮件已发送 -> {receiver}")
        return True
    except Exception as e:
        print(f"[notify] 邮件发送失败: {e}")
        return False


if __name__ == "__main__":
    send_email_html("测试邮件 - quant serve", "<p>notify 模块测试</p>")
