"""
周报生成脚本

使用方式:
    python -m analytics.weekly_report
    python -m analytics.weekly_report --weeks 2
    python -m analytics.weekly_report --output ./reports
"""

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

from analytics.event_analyzer import EventAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="生成溺水监测周报")
    parser.add_argument("--weeks", type=int, default=1, help="统计周数，默认为1周")
    parser.add_argument("--log-file", type=str, default="logs/events.jsonl", help="日志文件路径")
    parser.add_argument("--output", type=str, default="analytics/output", help="输出目录")
    args = parser.parse_args()
    
    logger.info(f"开始生成最近{args.weeks}周的周报...")
    
    # 创建分析器
    analyzer = EventAnalyzer(log_file=args.log_file)
    
    # 生成趋势图
    chart_path = analyzer.plot_daily_trend(days=7*args.weeks)
    
    # 生成报告
    report_path = analyzer.generate_report(
        report_type="weekly",
        output_dir=args.output
    )
    
    if report_path:
        logger.info(f"周报生成成功: {report_path}")
        
        # 打印统计摘要
        stats = analyzer.get_basic_stats()
        daily_stats = analyzer.get_daily_stats(days=7*args.weeks)
        
        print("\n" + "="*50)
        print("📊 周报摘要")
        print("="*50)
        print(f"统计周期: 最近{args.weeks}周")
        print(f"总事件数: {stats.get('total_events', 0)}")
        print(f"溺水报警: {stats.get('total_drowning_incidents', 0)}")
        
        if daily_stats:
            avg_daily = sum(d['total_drowning'] for d in daily_stats) / len(daily_stats)
            print(f"日均报警: {avg_daily:.1f}")
        
        print("="*50)
    else:
        logger.error("周报生成失败")


if __name__ == "__main__":
    main()
