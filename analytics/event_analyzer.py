"""
事件分析器 - 核心分析功能

提供事件数据的读取、统计和可视化功能
"""

import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# 配置日志
logger = logging.getLogger(__name__)

# 可选的可视化库
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib import rcParams
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib未安装，可视化功能不可用")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logger.warning("pandas未安装，部分分析功能受限")


class EventAnalyzer:
    """
    溺水检测事件分析器
    
    功能：
        - 读取和解析events.jsonl
        - 统计分析（时段、趋势、热点）
        - 生成可视化图表
        - 导出报告
    """
    
    def __init__(self, log_file: str = "logs/events.jsonl"):
        """
        初始化分析器
        
        Args:
            log_file: 事件日志文件路径
        """
        self.log_file = Path(log_file)
        self.events: List[Dict[str, Any]] = []
        self.df = None
        
        # 加载数据
        self._load_events()
    
    def _load_events(self) -> None:
        """加载事件数据"""
        if not self.log_file.exists():
            logger.warning(f"日志文件不存在: {self.log_file}")
            return
        
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                        self.events.append(event)
                    except json.JSONDecodeError:
                        continue
            
            logger.info(f"加载了 {len(self.events)} 条事件记录")
            
            # 转换为DataFrame（如果pandas可用）
            if PANDAS_AVAILABLE and self.events:
                self.df = pd.DataFrame(self.events)
                # 转换时间戳
                if 'time_iso' in self.df.columns:
                    self.df['timestamp'] = pd.to_datetime(self.df['time_iso'])
                
        except Exception as e:
            logger.error(f"加载事件数据失败: {e}")
    
    def get_basic_stats(self) -> Dict[str, Any]:
        """
        获取基本统计信息
        
        Returns:
            统计信息字典
        """
        if not self.events:
            return {"error": "无事件数据"}
        
        # 统计各类事件
        event_types = defaultdict(int)
        total_drowning = 0
        total_persons = 0
        
        for event in self.events:
            kind = event.get('kind', 'unknown')
            event_types[kind] += 1
            
            if kind == 'drowning_alert':
                total_drowning += event.get('drowning_count', 0)
                total_persons += event.get('person_count', 0)
        
        # 时间范围
        timestamps = [e.get('time_iso', '') for e in self.events if 'time_iso' in e]
        if timestamps:
            timestamps.sort()
            time_range = {
                "start": timestamps[0],
                "end": timestamps[-1],
                "duration_days": (pd.to_datetime(timestamps[-1]) - pd.to_datetime(timestamps[0])).days
                    if PANDAS_AVAILABLE else None
            }
        else:
            time_range = None
        
        return {
            "total_events": len(self.events),
            "event_types": dict(event_types),
            "total_drowning_incidents": total_drowning,
            "total_persons_detected": total_persons,
            "time_range": time_range
        }
    
    def get_hourly_distribution(self) -> Dict[str, List]:
        """
        获取小时级分布（溺水高发时段分析）
        
        Returns:
            每小时的事件数量
        """
        if not PANDAS_AVAILABLE or self.df is None:
            # 使用纯Python实现
            hourly_counts = defaultdict(int)
            for event in self.events:
                if event.get('kind') == 'drowning_alert':
                    time_str = event.get('time_local', '')
                    if time_str:
                        try:
                            hour = datetime.strptime(time_str[:19], "%Y-%m-%d %H:%M:%S").hour
                            hourly_counts[hour] += 1
                        except:
                            continue
            
            hours = list(range(24))
            counts = [hourly_counts.get(h, 0) for h in hours]
            return {"hours": hours, "counts": counts}
        
        # 使用pandas
        df_alerts = self.df[self.df['kind'] == 'drowning_alert'].copy()
        if df_alerts.empty:
            return {"hours": list(range(24)), "counts": [0] * 24}
        
        df_alerts['hour'] = df_alerts['timestamp'].dt.hour
        hourly = df_alerts.groupby('hour').size().reindex(range(24), fill_value=0)
        
        return {
            "hours": hourly.index.tolist(),
            "counts": hourly.values.tolist()
        }
    
    def get_daily_stats(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        获取最近N天的每日统计
        
        Args:
            days: 天数
            
        Returns:
            每日统计列表
        """
        if not PANDAS_AVAILABLE:
            logger.warning("pandas未安装，无法生成每日统计")
            return []
        
        if self.df is None or self.df.empty:
            return []
        
        # 过滤最近N天
        cutoff = datetime.now() - timedelta(days=days)
        recent = self.df[self.df['timestamp'] >= cutoff]
        
        if recent.empty:
            return []
        
        # 按天分组统计
        recent['date'] = recent['timestamp'].dt.date
        daily = recent.groupby('date').agg({
            'kind': 'count',
            'drowning_count': 'sum',
            'person_count': 'sum',
            'confidence': 'mean'
        }).reset_index()
        
        daily.columns = ['date', 'total_events', 'total_drowning', 'total_persons', 'avg_confidence']
        
        return daily.to_dict('records')
    
    def get_confidence_distribution(self) -> Dict[str, Any]:
        """
        获取置信度分布
        
        Returns:
            置信度统计信息
        """
        confidences = []
        for event in self.events:
            if event.get('kind') == 'drowning_alert':
                conf = event.get('confidence')
                if conf is not None:
                    confidences.append(conf)
        
        if not confidences:
            return {"error": "无置信度数据"}
        
        return {
            "count": len(confidences),
            "mean": np.mean(confidences),
            "median": np.median(confidences),
            "min": np.min(confidences),
            "max": np.max(confidences),
            "std": np.std(confidences)
        }
    
    def plot_hourly_distribution(self, save_path: Optional[str] = None) -> Optional[str]:
        """
        绘制小时级分布图（溺水高发时段）
        
        Args:
            save_path: 保存路径，默认为analytics/output/hourly_distribution.png
            
        Returns:
            保存的文件路径
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib未安装，无法绘图")
            return None
        
        data = self.get_hourly_distribution()
        hours = data['hours']
        counts = data['counts']
        
        # 设置中文字体
        rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
        rcParams['axes.unicode_minus'] = False
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars = ax.bar(hours, counts, color='#3498db', alpha=0.7, edgecolor='#2980b9')
        
        # 高亮最高值
        max_idx = np.argmax(counts)
        bars[max_idx].set_color('#e74c3c')
        
        ax.set_xlabel('小时 (Hour)', fontsize=12)
        ax.set_ylabel('溺水报警次数', fontsize=12)
        ax.set_title('溺水高发时段分析', fontsize=14, fontweight='bold')
        ax.set_xticks(hours)
        ax.set_xticklabels([f'{h:02d}:00' for h in hours], rotation=45)
        ax.grid(axis='y', alpha=0.3)
        
        # 添加数值标签
        for i, (h, c) in enumerate(zip(hours, counts)):
            if c > 0:
                ax.text(h, c + 0.5, str(c), ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        # 保存
        if save_path is None:
            output_dir = Path("analytics/output")
            output_dir.mkdir(parents=True, exist_ok=True)
            save_path = output_dir / "hourly_distribution.png"
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"图表已保存: {save_path}")
        return str(save_path)
    
    def plot_daily_trend(self, days: int = 7, save_path: Optional[str] = None) -> Optional[str]:
        """
        绘制每日趋势图
        
        Args:
            days: 天数
            save_path: 保存路径
            
        Returns:
            保存的文件路径
        """
        if not MATPLOTLIB_AVAILABLE or not PANDAS_AVAILABLE:
            logger.warning("matplotlib或pandas未安装，无法绘图")
            return None
        
        daily_stats = self.get_daily_stats(days)
        if not daily_stats:
            return None
        
        dates = [pd.to_datetime(d['date']) for d in daily_stats]
        counts = [d['total_drowning'] for d in daily_stats]
        
        rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
        rcParams['axes.unicode_minus'] = False
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(dates, counts, marker='o', linewidth=2, markersize=8, color='#e74c3c')
        ax.fill_between(dates, counts, alpha=0.3, color='#e74c3c')
        
        ax.set_xlabel('日期', fontsize=12)
        ax.set_ylabel('溺水事件数', fontsize=12)
        ax.set_title(f'最近{days}天溺水事件趋势', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 格式化日期
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path is None:
            output_dir = Path("analytics/output")
            output_dir.mkdir(parents=True, exist_ok=True)
            save_path = output_dir / f"daily_trend_{days}days.png"
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"趋势图已保存: {save_path}")
        return str(save_path)
    
    def generate_report(self, report_type: str = "daily", output_dir: str = "analytics/output") -> str:
        """
        生成分析报告
        
        Args:
            report_type: 报告类型 (daily, weekly, monthly)
            output_dir: 输出目录
            
        Returns:
            报告文件路径
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 生成统计
        basic_stats = self.get_basic_stats()
        hourly_data = self.get_hourly_distribution()
        confidence_stats = self.get_confidence_distribution()
        
        # 生成图表
        chart1 = self.plot_hourly_distribution()
        chart2 = None
        if report_type in ["weekly", "monthly"]:
            days = 7 if report_type == "weekly" else 30
            chart2 = self.plot_daily_trend(days=days)
        
        # 生成HTML报告
        report_file = output_path / f"{report_type}_report_{datetime.now().strftime('%Y%m%d')}.html"
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>溺水监测系统{report_type.upper()}报告</title>
    <style>
        body {{ font-family: 'Microsoft YaHei', sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .stat-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; text-align: center; }}
        .stat-value {{ font-size: 36px; font-weight: bold; }}
        .stat-label {{ font-size: 14px; opacity: 0.9; margin-top: 5px; }}
        .chart {{ margin: 30px 0; text-align: center; }}
        .chart img {{ max-width: 100%; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #3498db; color: white; }}
        tr:hover {{ background: #f5f5f5; }}
        .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #7f8c8d; font-size: 12px; text-align: center; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🌊 防溺水智能监测系统 - {report_type.upper()}报告</h1>
        <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>📊 基本统计</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{basic_stats.get('total_events', 0)}</div>
                <div class="stat-label">总事件数</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{basic_stats.get('total_drowning_incidents', 0)}</div>
                <div class="stat-label">溺水报警</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{basic_stats.get('total_persons_detected', 0)}</div>
                <div class="stat-label">检测人数</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{confidence_stats.get('mean', 0):.2%}</div>
                <div class="stat-label">平均置信度</div>
            </div>
        </div>
        
        <h2>⏰ 高发时段分析</h2>
        <div class="chart">
            <img src="hourly_distribution.png" alt="小时级分布">
        </div>
        
        <h2>📈 时段统计表</h2>
        <table>
            <tr><th>时段</th><th>报警次数</th><th>占比</th></tr>
"""
        
        total = sum(hourly_data['counts'])
        for hour, count in zip(hourly_data['hours'], hourly_data['counts']):
            if count > 0:
                percentage = count / total * 100 if total > 0 else 0
                html_content += f"<tr><td>{hour:02d}:00 - {hour+1:02d}:00</td><td>{count}</td><td>{percentage:.1f}%</td></tr>"
        
        html_content += f"""
        </table>
        
        {f'<h2>📉 趋势分析</h2><div class="chart"><img src="daily_trend_{7 if report_type == "weekly" else 30}days.png" alt="趋势图"></div>' if chart2 else ''}
        
        <div class="footer">
            <p>防溺水智能监测系统 v2.0 | 数据科学与大数据技术毕业设计</p>
        </div>
    </div>
</body>
</html>
"""
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"报告已生成: {report_file}")
        return str(report_file)
