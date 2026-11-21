同步 WALLX harness

使用飞书《黑客松选手指南》中的任务导入、WebUI 和 harness 步骤。

安装：python -m pip install -r requirements.txt
检查：python -m unittest -v test_harness.py
转发：python harness.py --host YOUR_LAN_IP --port 8001 --passthrough
规划：设置 AGENT_CHAT_URL、AGENT_MODEL、AGENT_API_KEY 后去掉 --passthrough。

模型地址由 WALLX_ENDPOINT 指定。握手鉴权使用 WALLX_HEADERS_FILE 指向的私有 JSON。
client 使用 sync、public_v2 和与工位匹配的 embodiment。停止通过 WebUI 完成。
