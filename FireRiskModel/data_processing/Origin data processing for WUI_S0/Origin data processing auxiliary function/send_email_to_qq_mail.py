# -- coding: utf-8 --
# ==============================================================================
# 邮件发送脚本（简化版）
# 功能：使用Python发送电子邮件到指定邮箱
# 注意事项：
#   1. 使用前需要配置发件人邮箱、密码和SMTP服务器信息
#   2. 对于QQ邮箱，需要使用授权码而不是登录密码
#   3. 某些邮件服务器可能需要开启SMTP服务
# ==============================================================================

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header


def send_email(
    recipient_email="1991860410@qq.com",
    subject="测试邮件",
    content="这是一封测试邮件，由Python自动发送。",
    sender_email="your_email@example.com",
    smtp_server="smtp.example.com",
    smtp_port=587,
    smtp_username="your_email@example.com",
    smtp_password="your_password_or_app_password"
):
    """
    发送电子邮件
    
    Args:
        recipient_email: 收件人邮箱地址
        subject: 邮件主题
        content: 邮件内容
        sender_email: 发件人邮箱地址
        smtp_server: SMTP服务器地址
        smtp_port: SMTP服务器端口
        smtp_username: SMTP用户名
        smtp_password: SMTP密码或应用授权码
    
    Returns:
        bool: 发送成功返回True，失败返回False
    """
    # 创建邮件对象
    message = MIMEMultipart()
    message['From'] = Header(sender_email)
    message['To'] = Header(recipient_email)
    message['Subject'] = Header(subject, 'utf-8')
    
    # 添加邮件正文
    message.attach(MIMEText(content, 'plain', 'utf-8'))
    
    # 连接SMTP服务器并发送邮件
    print("正在连接SMTP服务器 {0}:{1}...".format(smtp_server, smtp_port))
    server = smtplib.SMTP(smtp_server, smtp_port)
    
    # 启用TLS加密
    server.starttls()
    
    # 登录SMTP服务器
    print("正在登录SMTP服务器...")
    server.login(smtp_username, smtp_password)
    
    # 发送邮件
    print("正在发送邮件到 {0}...".format(recipient_email))
    server.sendmail(sender_email, recipient_email, message.as_string())
    
    # 关闭连接
    server.quit()
    
    print("邮件发送成功！")
    return True


def send_qq_email(
    recipient_email="1991860410@qq.com",
    subject="测试邮件",
    content="这是一封测试邮件，由Python自动发送。",
    sender_email="your_qq_email@qq.com",
    smtp_password="your_qq_app_password"
):
    """
    使用QQ邮箱发送邮件（简化版本）
    
    Args:
        recipient_email: 收件人邮箱地址
        subject: 邮件主题
        content: 邮件内容
        sender_email: 发件人QQ邮箱地址
        smtp_password: QQ邮箱授权码（非登录密码）
    
    Returns:
        bool: 发送成功返回True，失败返回False
    """
    # QQ邮箱SMTP服务器配置
    return send_email(
        recipient_email=recipient_email,
        subject=subject,
        content=content,
        sender_email=sender_email,
        smtp_server="smtp.qq.com",
        smtp_port=587,
        smtp_username=sender_email,
        smtp_password=smtp_password
    )


if __name__ == "__main__":
    # 使用说明：
    # 1. 请先在QQ邮箱中开启SMTP服务并获取授权码
    # 2. 将下面的发件人邮箱和授权码替换为实际的值
    # 3. 运行脚本即可发送邮件
    
    # 配置信息（已填入用户提供的值）
    sender_email = "1991860410@qq.com"  # 用户的QQ邮箱
    app_password = "uflzfzvzcbyddbgc"  # 用户提供的QQ邮箱授权码
    
    # 发送邮件
    success = send_qq_email(
        recipient_email="1991860410@qq.com",
        subject="测试邮件",
        content="这是一封测试邮件，由Python自动发送。\n\n这是邮件的第二行内容。",
        sender_email=sender_email,
        smtp_password=app_password
    )
    
    # 发送结果
    if success:
        print("任务完成：邮件已成功发送！")