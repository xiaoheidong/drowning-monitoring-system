"""
大数据组件模块 - 可选扩展功能

本模块提供大数据处理能力，包括：
- Elasticsearch事件存储与检索
- Kafka消息队列（预留接口）
- Spark Streaming处理（预留接口）

使用方式：
    在config/settings.json中设置 enable_elasticsearch: true 启用ES功能

注意：
    本模块为可选功能，不启用时不会影响主程序运行
"""

__version__ = "1.0.0"
