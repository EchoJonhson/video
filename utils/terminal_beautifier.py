#!/usr/bin/env python3
"""
终端美化工具 - 提供彩色输出和进度条显示
"""

import os
import sys
import time
import threading
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Callable

# ANSI 转义码定义
class Colors:
    """终端颜色定义"""
    # 基础颜色
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # 明亮颜色
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'
    
    # 背景色
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'
    
    # 样式
    BOLD = '\033[1m'
    DIM = '\033[2m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'
    BLINK = '\033[5m'
    REVERSE = '\033[7m'
    HIDDEN = '\033[8m'
    STRIKETHROUGH = '\033[9m'
    
    # 重置
    RESET = '\033[0m'
    
    # 组合样式
    HEADER = BOLD + BRIGHT_CYAN
    SUCCESS = BRIGHT_GREEN
    ERROR = BRIGHT_RED
    WARNING = BRIGHT_YELLOW
    INFO = BRIGHT_BLUE
    PROCESS = BRIGHT_MAGENTA
    FILE = BRIGHT_YELLOW
    TIME = BRIGHT_BLACK


class ProgressBar:
    """终端进度条"""
    
    def __init__(self, total: int, width: int = 50, title: str = ""):
        self.total = total
        self.width = width
        self.title = title
        self.current = 0
        self.start_time = time.time()
        self.last_update = time.time()
        self.smooth_eta = 0
        self.speed_history = []
        
    def update(self, current: int, status: str = ""):
        """更新进度条"""
        self.current = current
        progress = current / self.total if self.total > 0 else 0
        filled = int(self.width * progress)
        
        # 计算时间和速度
        elapsed = time.time() - self.start_time
        current_time = time.time()
        
        # 计算瞬时速度（过去1秒的速度）
        if current > 0:
            time_diff = current_time - self.last_update
            if time_diff > 0.1:  # 每0.1秒更新一次速度
                speed = 1 / time_diff
                self.speed_history.append(speed)
                if len(self.speed_history) > 10:  # 保留最近10个速度样本
                    self.speed_history.pop(0)
                self.last_update = current_time
                
        # 计算平滑的ETA
        if self.speed_history and progress > 0:
            avg_speed = sum(self.speed_history) / len(self.speed_history)
            remaining = self.total - current
            eta = remaining / avg_speed if avg_speed > 0 else 0
            self.smooth_eta = 0.7 * self.smooth_eta + 0.3 * eta  # 指数移动平均
        else:
            self.smooth_eta = 0
            
        # 构建渐变进度条
        bar_chars = []
        for i in range(self.width):
            if i < filled:
                # 使用不同颜色创建渐变效果
                if i < filled * 0.25:
                    bar_chars.append(f"{Colors.RED}█{Colors.RESET}")
                elif i < filled * 0.5:
                    bar_chars.append(f"{Colors.YELLOW}█{Colors.RESET}")
                elif i < filled * 0.75:
                    bar_chars.append(f"{Colors.BLUE}█{Colors.RESET}")
                else:
                    bar_chars.append(f"{Colors.GREEN}█{Colors.RESET}")
            else:
                bar_chars.append(f"{Colors.DIM}░{Colors.RESET}")
        bar = ''.join(bar_chars)
        
        # 添加动画效果
        spinner = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'][int(time.time() * 10) % 10]
        
        # 清除当前行并重新打印
        print(f"\r{spinner} {Colors.BOLD}{self.title}{Colors.RESET} [{bar}] "
              f"{Colors.BRIGHT_GREEN}{progress*100:5.1f}%{Colors.RESET} "
              f"({Colors.BRIGHT_CYAN}{current}{Colors.RESET}/{self.total}) "
              f"⏱️  {Colors.TIME}{self._format_time(elapsed)}{Colors.RESET} "
              f"ETA: {Colors.BRIGHT_YELLOW}{self._format_time(self.smooth_eta)}{Colors.RESET} "
              f"{Colors.DIM}{status}{Colors.RESET}", end='', flush=True)
        
        if current >= self.total:
            print(f"\r✨ {Colors.BOLD}{self.title}{Colors.RESET} [{bar}] "
                  f"{Colors.BRIGHT_GREEN}100.0%{Colors.RESET} - "
                  f"{Colors.SUCCESS}完成！{Colors.RESET} "
                  f"总耗时: {Colors.BRIGHT_CYAN}{self._format_time(elapsed)}{Colors.RESET}")
    
    def _format_time(self, seconds: float) -> str:
        """格式化时间显示"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            mins = int(seconds / 60)
            secs = int(seconds % 60)
            return f"{mins}m{secs}s"
        else:
            hours = int(seconds / 3600)
            mins = int((seconds % 3600) / 60)
            return f"{hours}h{mins}m"


class Spinner:
    """动画旋转器"""
    
    def __init__(self, message: str = "处理中"):
        self.message = message
        self.spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        self.running = False
        self.thread = None
        self.current = 0
        
    def start(self):
        """开始动画"""
        self.running = True
        self.thread = threading.Thread(target=self._animate)
        self.thread.daemon = True
        self.thread.start()
        
    def _animate(self):
        """动画循环"""
        while self.running:
            char = self.spinner_chars[self.current % len(self.spinner_chars)]
            print(f"\r{char} {Colors.INFO}{self.message}{Colors.RESET}", end='', flush=True)
            self.current += 1
            time.sleep(0.1)
            
    def stop(self, success_msg: str = None, error_msg: str = None):
        """停止动画"""
        self.running = False
        if self.thread:
            self.thread.join()
        
        # 清除当前行
        print('\r' + ' ' * (len(self.message) + 10), end='', flush=True)
        
        if success_msg:
            print(f"\r✅ {Colors.SUCCESS}{success_msg}{Colors.RESET}")
        elif error_msg:
            print(f"\r❌ {Colors.ERROR}{error_msg}{Colors.RESET}")
        else:
            print("\r", end='')


class TerminalBeautifier:
    """终端美化器"""
    
    def __init__(self, enable_color: bool = True):
        self.enable_color = enable_color and self._supports_color()
        self.spinners = []  # 跟踪所有活动的旋转器
        
    def _supports_color(self) -> bool:
        """检查终端是否支持颜色"""
        # Windows 终端检查
        if sys.platform == 'win32':
            return os.environ.get('ANSICON') is not None or 'WT_SESSION' in os.environ
        # Unix/Linux 终端检查
        return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
    
    def print_header(self, title: str, subtitle: str = ""):
        """打印美化的标题"""
        width = 80
        # 使用更美观的边框
        print(f"\n{Colors.BRIGHT_CYAN}{'━' * width}{Colors.RESET}")
        print(f"{Colors.HEADER}✨ {title.center(width-4)} ✨{Colors.RESET}")
        if subtitle:
            print(f"{Colors.DIM}{subtitle.center(width)}{Colors.RESET}")
        print(f"{Colors.BRIGHT_CYAN}{'━' * width}{Colors.RESET}\n")
    
    def print_section(self, title: str, icon: str = "📌"):
        """打印章节标题"""
        # 计算实际显示长度（考虑中文字符）
        display_len = sum(2 if ord(c) > 127 else 1 for c in title) + 4
        print(f"\n{icon} {Colors.BOLD}{Colors.BRIGHT_CYAN}{title}{Colors.RESET}")
        print(f"{Colors.BRIGHT_CYAN}{'─' * display_len}{Colors.RESET}")
    
    def print_step(self, step_num: int, total_steps: int, title: str):
        """打印步骤信息"""
        # 添加进度指示器
        progress = step_num / total_steps
        progress_bar = self._mini_progress_bar(progress, 10)
        print(f"\n{Colors.BOLD}[步骤 {step_num}/{total_steps}]{Colors.RESET} {progress_bar} "
              f"{Colors.PROCESS}{title}{Colors.RESET}")
    
    def print_info(self, message: str, icon: str = "ℹ️"):
        """打印信息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"{Colors.DIM}[{timestamp}]{Colors.RESET} {icon}  {Colors.INFO}{message}{Colors.RESET}")
    
    def print_success(self, message: str, icon: str = "✅"):
        """打印成功信息"""
        print(f"{icon} {Colors.SUCCESS}{message}{Colors.RESET}")
    
    def print_error(self, message: str, icon: str = "❌"):
        """打印错误信息"""
        print(f"{icon} {Colors.ERROR}{message}{Colors.RESET}")
    
    def print_warning(self, message: str, icon: str = "⚠️"):
        """打印警告信息"""
        print(f"{icon}  {Colors.WARNING}{message}{Colors.RESET}")
    
    def print_file(self, filename: str, size_mb: float, icon: str = "📄"):
        """打印文件信息"""
        print(f"{icon} {Colors.FILE}{filename}{Colors.RESET} "
              f"{Colors.DIM}({size_mb:.2f} MB){Colors.RESET}")
    
    def print_stats(self, stats: Dict[str, Any]):
        """打印统计信息"""
        print(f"\n{Colors.BOLD}📊 统计信息{Colors.RESET}")
        print(f"{Colors.DIM}{'─' * 40}{Colors.RESET}")
        
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {Colors.BRIGHT_GREEN}{value:.2f}{Colors.RESET}")
            elif isinstance(value, int):
                print(f"  {key}: {Colors.BRIGHT_GREEN}{value:,}{Colors.RESET}")
            elif isinstance(value, timedelta):
                print(f"  {key}: {Colors.BRIGHT_GREEN}{str(value)}{Colors.RESET}")
            else:
                print(f"  {key}: {Colors.BRIGHT_GREEN}{value}{Colors.RESET}")
    
    def print_table(self, headers: List[str], rows: List[List[str]]):
        """打印表格"""
        # 计算列宽
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(cell)))
        
        # 打印表头
        header_line = " │ ".join(f"{h:<{w}}" for h, w in zip(headers, col_widths))
        print(f"\n{Colors.BOLD}{header_line}{Colors.RESET}")
        print(f"{Colors.DIM}{'─' * len(header_line)}{Colors.RESET}")
        
        # 打印行
        for row in rows:
            row_line = " │ ".join(f"{str(cell):<{w}}" for cell, w in zip(row, col_widths))
            print(row_line)
    
    def print_box(self, title: str, content: List[str], width: int = 60, style: str = "default"):
        """打印带框的内容"""
        # 支持不同风格的边框
        if style == "double":
            tl, tr, bl, br = "╔", "╗", "╚", "╝"
            h, v = "═", "║"
            ml, mr = "╠", "╣"
        else:
            tl, tr, bl, br = "┌", "┐", "└", "┘"
            h, v = "─", "│"
            ml, mr = "├", "┤"
            
        print(f"\n{Colors.BRIGHT_CYAN}{tl}{h * (width - 2)}{tr}{Colors.RESET}")
        print(f"{Colors.BRIGHT_CYAN}{v}{Colors.RESET} {Colors.BOLD}{title.center(width - 4)}{Colors.RESET} {Colors.BRIGHT_CYAN}{v}{Colors.RESET}")
        print(f"{Colors.BRIGHT_CYAN}{ml}{h * (width - 2)}{mr}{Colors.RESET}")
        
        for line in content:
            # 处理中文字符宽度
            display_len = sum(2 if ord(c) > 127 else 1 for c in line)
            padding = width - 4 - display_len
            print(f"{Colors.BRIGHT_CYAN}{v}{Colors.RESET} {line}{' ' * max(0, padding)} {Colors.BRIGHT_CYAN}{v}{Colors.RESET}")
        
        print(f"{Colors.BRIGHT_CYAN}{bl}{h * (width - 2)}{br}{Colors.RESET}")
    
    def create_progress_bar(self, total: int, title: str = "") -> ProgressBar:
        """创建进度条"""
        return ProgressBar(total, title=title)
    
    def print_model_config(self, model_type: str, config: Dict[str, Any]):
        """打印模型配置信息"""
        print(f"\n{Colors.BOLD}🤖 模型配置{Colors.RESET}")
        print(f"{Colors.DIM}{'─' * 40}{Colors.RESET}")
        print(f"  模型类型: {Colors.BRIGHT_MAGENTA}{model_type.upper()}{Colors.RESET}")
        
        for key, value in config.items():
            if key == "max_workers":
                print(f"  并行线程: {Colors.BRIGHT_GREEN}{value}{Colors.RESET}")
            elif key == "batch_size":
                print(f"  批处理大小: {Colors.BRIGHT_GREEN}{value}{Colors.RESET}")
            elif key == "memory_usage":
                print(f"  内存使用: {Colors.BRIGHT_YELLOW}{value:.1f}GB{Colors.RESET}")
    
    def print_hardware_info(self, cpu_model: str, cpu_cores: int, memory_gb: float):
        """打印硬件信息"""
        print(f"\n{Colors.BOLD}💻 硬件配置{Colors.RESET}")
        print(f"{Colors.DIM}{'─' * 40}{Colors.RESET}")
        print(f"  CPU: {Colors.BRIGHT_CYAN}{cpu_model}{Colors.RESET}")
        print(f"  核心数: {Colors.BRIGHT_GREEN}{cpu_cores}{Colors.RESET}")
        print(f"  内存: {Colors.BRIGHT_GREEN}{memory_gb:.1f}GB{Colors.RESET}")
    
    def _mini_progress_bar(self, progress: float, width: int = 10) -> str:
        """创建迷你进度条"""
        filled = int(width * progress)
        bar = "▰" * filled + "▱" * (width - filled)
        return f"{Colors.BRIGHT_GREEN}{bar}{Colors.RESET}"
    
    def create_spinner(self, message: str) -> Spinner:
        """创建动画旋转器"""
        spinner = Spinner(message)
        self.spinners.append(spinner)
        return spinner
    
    def print_processing_status(self, current: int, total: int, item_name: str, status: str = "处理中"):
        """打印处理状态（适用于批量处理）"""
        progress = current / total if total > 0 else 0
        bar = self._mini_progress_bar(progress, 20)
        
        # 选择状态图标
        if status == "处理中":
            icon = "🔄"
            color = Colors.INFO
        elif status == "完成":
            icon = "✅"
            color = Colors.SUCCESS
        elif status == "失败":
            icon = "❌"
            color = Colors.ERROR
        else:
            icon = "⏳"
            color = Colors.WARNING
            
        print(f"\r[{current}/{total}] {bar} {icon} {color}{item_name}{Colors.RESET} - {status}", 
              end='', flush=True)
        
        if current >= total:
            print()  # 完成时换行
    
    def print_summary(self, title: str, stats: Dict[str, Any], style: str = "double"):
        """打印美化的总结信息"""
        # 准备内容
        content = []
        for key, value in stats.items():
            if isinstance(value, float):
                line = f"{key}: {value:.2f}"
            elif isinstance(value, int):
                line = f"{key}: {value:,}"
            else:
                line = f"{key}: {value}"
            content.append(line)
        
        # 使用双线框显示总结
        self.print_box(title, content, width=60, style=style)
    
    def print_file_processing(self, filename: str, current_size: int, total_size: int):
        """打印文件处理进度"""
        progress = current_size / total_size if total_size > 0 else 0
        size_mb = total_size / 1024 / 1024
        current_mb = current_size / 1024 / 1024
        
        # 速度计算
        if not hasattr(self, '_file_start_time'):
            self._file_start_time = time.time()
            self._last_size = 0
        
        elapsed = time.time() - self._file_start_time
        if elapsed > 0:
            speed = (current_size - self._last_size) / elapsed / 1024 / 1024  # MB/s
            self._last_size = current_size
        else:
            speed = 0
            
        bar = self._mini_progress_bar(progress, 30)
        print(f"\r📁 {Colors.FILE}{filename}{Colors.RESET} {bar} "
              f"{current_mb:.1f}/{size_mb:.1f}MB ({progress*100:.1f}%) "
              f"速度: {Colors.BRIGHT_GREEN}{speed:.1f}MB/s{Colors.RESET}", 
              end='', flush=True)


# 全局实例
beautifier = TerminalBeautifier()


# 便捷函数
def print_header(title: str, subtitle: str = ""):
    beautifier.print_header(title, subtitle)

def print_section(title: str, icon: str = "📌"):
    beautifier.print_section(title, icon)

def print_step(step_num: int, total_steps: int, title: str):
    beautifier.print_step(step_num, total_steps, title)

def print_info(message: str, icon: str = "ℹ️"):
    beautifier.print_info(message, icon)

def print_success(message: str, icon: str = "✅"):
    beautifier.print_success(message, icon)

def print_error(message: str, icon: str = "❌"):
    beautifier.print_error(message, icon)

def print_warning(message: str, icon: str = "⚠️"):
    beautifier.print_warning(message, icon)

def create_progress_bar(total: int, title: str = "") -> ProgressBar:
    return beautifier.create_progress_bar(total, title)


if __name__ == "__main__":
    # 测试代码
    print_header("FireRedASR 终端美化测试", "Terminal Beautifier Demo")
    
    print_section("基础颜色测试")
    print_info("这是一条信息")
    print_success("这是成功消息")
    print_warning("这是警告消息")
    print_error("这是错误消息")
    
    print_section("进度条测试")
    progress = create_progress_bar(100, "处理进度")
    for i in range(101):
        progress.update(i, f"处理文件 {i}")
        time.sleep(0.01)
    
    print_section("表格测试")
    beautifier.print_table(
        ["文件名", "大小", "状态"],
        [
            ["video1.mp4", "125.3MB", "✅ 完成"],
            ["video2.mp4", "89.7MB", "🔄 处理中"],
            ["video3.mp4", "156.2MB", "⏳ 等待"],
        ]
    )
    
    print("\n✨ 测试完成！")