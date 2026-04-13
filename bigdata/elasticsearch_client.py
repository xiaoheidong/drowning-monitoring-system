"""
Elasticsearch客户端 - 可选的事件存储与检索功能

功能：
    - 将溺水检测事件写入Elasticsearch
    - 提供事件检索和统计分析接口
    - 支持全文搜索和聚合查询

使用方式：
    1. 安装Elasticsearch并启动服务
    2. 在config/settings.json中配置：
       {
           "elasticsearch": {
               "enabled": true,
               "host": "localhost",
               "port": 9200,
               "index": "drowning-events"
           }
       }
    3. EventLogger会自动检测并使用ES存储

依赖安装：
    pip install elasticsearch>=8.0.0
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

# 配置日志
logger = logging.getLogger(__name__)

# 尝试导入elasticsearch，如果未安装则给出友好提示
try:
    from elasticsearch import Elasticsearch
    from elasticsearch.helpers import bulk
    ELASTICSEARCH_AVAILABLE = True
except ImportError:
    ELASTICSEARCH_AVAILABLE = False
    logger.warning("elasticsearch包未安装，ES功能不可用。运行: pip install elasticsearch>=8.0.0")


class ElasticsearchClient:
    """
    Elasticsearch客户端封装类
    
    提供事件数据的存储、检索和统计分析功能。
    如果ES服务不可用，会自动降级为仅使用本地日志。
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 9200,
        index_name: str = "drowning-events",
        enabled: bool = True
    ):
        """
        初始化ES客户端
        
        Args:
            host: ES服务器地址
            port: ES端口
            index_name: 索引名称
            enabled: 是否启用ES功能
        """
        self.host = host
        self.port = port
        self.index_name = index_name
        self.enabled = enabled and ELASTICSEARCH_AVAILABLE
        self.client: Optional[Elasticsearch] = None
        
        if not self.enabled:
            logger.info("Elasticsearch功能已禁用或未安装")
            return
            
        if not ELASTICSEARCH_AVAILABLE:
            logger.warning("elasticsearch包未安装，无法启用ES功能")
            self.enabled = False
            return
        
        try:
            self.client = Elasticsearch([f"http://{host}:{port}"])
            # 测试连接
            if self.client.ping():
                logger.info(f"成功连接到Elasticsearch: {host}:{port}")
                self._ensure_index()
            else:
                logger.warning("无法连接到Elasticsearch服务，将使用本地日志")
                self.enabled = False
        except Exception as e:
            logger.warning(f"Elasticsearch连接失败: {e}，将使用本地日志")
            self.enabled = False
    
    def _ensure_index(self) -> None:
        """确保索引存在，如果不存在则创建"""
        if not self.client or not self.enabled:
            return
            
        try:
            if not self.client.indices.exists(index=self.index_name):
                # 创建索引并设置映射
                mapping = {
                    "mappings": {
                        "properties": {
                            "timestamp": {"type": "date"},
                            "event_type": {"type": "keyword"},
                            "person_count": {"type": "integer"},
                            "drowning_count": {"type": "integer"},
                            "confidence": {"type": "float"},
                            "location": {"type": "geo_point"},  # 预留地理位置字段
                            "screenshot_path": {"type": "text"},
                            "clip_path": {"type": "text"},
                            "note": {"type": "text"}
                        }
                    },
                    "settings": {
                        "number_of_shards": 1,
                        "number_of_replicas": 0
                    }
                }
                self.client.indices.create(index=self.index_name, body=mapping)
                logger.info(f"创建索引: {self.index_name}")
        except Exception as e:
            logger.error(f"创建索引失败: {e}")
    
    def index_event(self, event_data: Dict[str, Any]) -> bool:
        """
        索引单个事件
        
        Args:
            event_data: 事件数据字典
            
        Returns:
            是否成功
        """
        if not self.enabled or not self.client:
            return False
        
        try:
            # 添加时间戳
            if "timestamp" not in event_data:
                event_data["timestamp"] = datetime.now().isoformat()
            
            self.client.index(index=self.index_name, document=event_data)
            return True
        except Exception as e:
            logger.error(f"索引事件失败: {e}")
            return False
    
    def search_events(
        self,
        query: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        event_type: Optional[str] = None,
        size: int = 100
    ) -> List[Dict[str, Any]]:
        """
        搜索事件
        
        Args:
            query: 全文搜索关键词
            start_time: 开始时间 (ISO格式)
            end_time: 结束时间 (ISO格式)
            event_type: 事件类型筛选
            size: 返回结果数量
            
        Returns:
            事件列表
        """
        if not self.enabled or not self.client:
            return []
        
        try:
            must_conditions = []
            
            # 全文搜索
            if query:
                must_conditions.append({
                    "multi_match": {
                        "query": query,
                        "fields": ["note", "event_type"]
                    }
                })
            
            # 时间范围
            if start_time or end_time:
                time_range = {}
                if start_time:
                    time_range["gte"] = start_time
                if end_time:
                    time_range["lte"] = end_time
                must_conditions.append({
                    "range": {"timestamp": time_range}
                })
            
            # 事件类型
            if event_type:
                must_conditions.append({
                    "term": {"event_type": event_type}
                })
            
            # 构建查询
            search_body = {
                "size": size,
                "sort": [{"timestamp": {"order": "desc"}}],
                "query": {
                    "bool": {
                        "must": must_conditions if must_conditions else [{"match_all": {}}]
                    }
                }
            }
            
            response = self.client.search(index=self.index_name, body=search_body)
            hits = response["hits"]["hits"]
            
            return [{"id": hit["_id"], **hit["_source"]} for hit in hits]
        except Exception as e:
            logger.error(f"搜索事件失败: {e}")
            return []
    
    def get_hourly_stats(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        获取小时级统计
        
        Args:
            hours: 最近多少小时
            
        Returns:
            每小时的事件统计
        """
        if not self.enabled or not self.client:
            return []
        
        try:
            search_body = {
                "size": 0,
                "query": {
                    "range": {
                        "timestamp": {
                            "gte": f"now-{hours}h"
                        }
                    }
                },
                "aggs": {
                    "hourly": {
                        "date_histogram": {
                            "field": "timestamp",
                            "calendar_interval": "hour",
                            "format": "yyyy-MM-dd HH:mm"
                        },
                        "aggs": {
                            "total_events": {"value_count": {"field": "_id"}},
                            "avg_confidence": {"avg": {"field": "confidence"}},
                            "sum_drowning": {"sum": {"field": "drowning_count"}}
                        }
                    }
                }
            }
            
            response = self.client.search(index=self.index_name, body=search_body)
            buckets = response["aggregations"]["hourly"]["buckets"]
            
            return [
                {
                    "time": bucket["key_as_string"],
                    "count": bucket["total_events"]["value"],
                    "avg_confidence": bucket["avg_confidence"]["value"],
                    "total_drowning": bucket["sum_drowning"]["value"]
                }
                for bucket in buckets
            ]
        except Exception as e:
            logger.error(f"获取统计失败: {e}")
            return []
    
    def get_daily_report(self, date: Optional[str] = None) -> Dict[str, Any]:
        """
        获取日报数据
        
        Args:
            date: 日期 (YYYY-MM-DD)，默认为今天
            
        Returns:
            日报统计信息
        """
        if not self.enabled or not self.client:
            return {}
        
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        try:
            search_body = {
                "size": 0,
                "query": {
                    "range": {
                        "timestamp": {
                            "gte": f"{date}T00:00:00",
                            "lte": f"{date}T23:59:59"
                        }
                    }
                },
                "aggs": {
                    "total_events": {"value_count": {"field": "_id"}},
                    "total_drowning": {"sum": {"field": "drowning_count"}},
                    "avg_confidence": {"avg": {"field": "confidence"}},
                    "hourly_distribution": {
                        "date_histogram": {
                            "field": "timestamp",
                            "calendar_interval": "hour"
                        }
                    }
                }
            }
            
            response = self.client.search(index=self.index_name, body=search_body)
            aggs = response["aggregations"]
            
            return {
                "date": date,
                "total_events": aggs["total_events"]["value"],
                "total_drowning": aggs["total_drowning"]["value"],
                "avg_confidence": aggs["avg_confidence"]["value"],
                "hourly_distribution": [
                    {"hour": bucket["key_as_string"], "count": bucket["doc_count"]}
                    for bucket in aggs["hourly_distribution"]["buckets"]
                ]
            }
        except Exception as e:
            logger.error(f"获取日报失败: {e}")
            return {}
    
    def close(self) -> None:
        """关闭ES连接"""
        if self.client:
            self.client.close()
            logger.info("Elasticsearch连接已关闭")


# 便捷函数：从settings创建客户端
def create_es_client_from_settings(settings: Dict[str, Any]) -> ElasticsearchClient:
    """
    从配置字典创建ES客户端
    
    Args:
        settings: 配置字典，包含elasticsearch配置
        
    Returns:
        ElasticsearchClient实例
    """
    es_config = settings.get("elasticsearch", {})
    return ElasticsearchClient(
        host=es_config.get("host", "localhost"),
        port=es_config.get("port", 9200),
        index_name=es_config.get("index", "drowning-events"),
        enabled=es_config.get("enabled", False)
    )
