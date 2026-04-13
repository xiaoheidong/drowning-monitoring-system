"""
日报生成脚本

使用方式:
    python -m analytics.daily_report
    python -m analytics.daily_report --date 2026-04-14
    python -m analytics.daily_report --output ./reports
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

from analytics.event_analyzer import EventAnalyzer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="生成溺水监测日报")
    parser.add_argument("--date", type=str, help="日期 (YYYY-MM-DD)，默认为今天")
    parser.add_argument("--log-file", type=str, default="logs/events.jsonl", help="日志文件路径")
    parser.add_argument("--output", type=str, default="analytics/output", help="输出目录")
    args = parser.parse_args()
    
    logger.info("开始生成日报...")
    
    # 创建分析器
    analyzer = EventAnalyzer(log_file=args.log_file)
    
    # 生成报告
    report_path = analyzer.generate_report(
        report_type="daily",
        output_dir=args.output
    )
    
    if report_path:
        logger.info(f"日报生成成功: {report_path}")
        
        # 打印统计摘要
        stats = analyzer.get_basic_stats()
        print("\n" + "="*50)
        print("📊 日报摘要")
        print("="*50)
        print(f"总事件数: {stats.get('total_events', 0)}")
        print(f"溺水报警: {stats.get('total_drowning_incidents', 0)}")
        print(f"检测人数: {stats.get('total_persons_detected', 0)}")
        
        hourly = analyzer.get_hourly_distribution()
        if hourly['counts']:
            max_hour = hourly['hours'][hourly['counts'].index(max(hourly['counts']))]
            print(f"高发时段: {max_hour:02d}:00-{max_hour+1:02d}:00")
        print("="*50)
    else:
        logger.error("日报生成失败")


if __name__ == "__main__":
    main()
