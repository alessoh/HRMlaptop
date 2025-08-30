# monitor.py - Monitor system resources during training
"""
Monitor CPU and memory usage during HRM training on Windows.
Run this in a separate terminal while training.
"""

import psutil
import matplotlib.pyplot as plt
from datetime import datetime
import time
import json
import os

class WindowsMonitor:
    def __init__(self):
        self.cpu_history = []
        self.memory_history = []
        self.time_stamps = []
        self.start_time = time.time()
        
        # Get system info
        self.system_info = {
            'cpu_count': psutil.cpu_count(logical=False),
            'cpu_threads': psutil.cpu_count(logical=True),
            'total_memory_gb': round(psutil.virtual_memory().total / (1024**3), 1),
            'cpu_freq_mhz': psutil.cpu_freq().current if psutil.cpu_freq() else 0,
            'platform': os.name
        }
        
        print("System Monitor Started")
        print(f"CPU: {self.system_info['cpu_count']} cores, {self.system_info['cpu_threads']} threads")
        print(f"RAM: {self.system_info['total_memory_gb']} GB")
        print("-" * 40)
        
    def monitor_step(self):
        """Record current system state"""
        current_time = time.time() - self.start_time
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        
        self.time_stamps.append(current_time)
        self.cpu_history.append(cpu_percent)
        self.memory_history.append(memory_percent)
        
        # Print current stats
        print(f"[{datetime.now().strftime('%H:%M:%S')}] "
              f"CPU: {cpu_percent:5.1f}% | "
              f"Memory: {memory_percent:5.1f}% | "
              f"Time: {current_time/60:.1f} min")
        
        return cpu_percent, memory_percent
    
    def run(self, duration_minutes=60, interval_seconds=5):
        """Run monitoring for specified duration"""
        print(f"\nMonitoring for {duration_minutes} minutes...")
        print("Press Ctrl+C to stop early\n")
        
        duration_seconds = duration_minutes * 60
        steps = int(duration_seconds / interval_seconds)
        
        try:
            for _ in range(steps):
                self.monitor_step()
                time.sleep(interval_seconds)
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
        
        self.save_results()
        self.plot_results()
        
    def save_results(self):
        """Save monitoring data to file"""
        results = {
            'system_info': self.system_info,
            'monitoring_data': {
                'time_stamps': self.time_stamps,
                'cpu_usage': self.cpu_history,
                'memory_usage': self.memory_history
            },
            'summary': {
                'avg_cpu': sum(self.cpu_history) / len(self.cpu_history) if self.cpu_history else 0,
                'max_cpu': max(self.cpu_history) if self.cpu_history else 0,
                'avg_memory': sum(self.memory_history) / len(self.memory_history) if self.memory_history else 0,
                'max_memory': max(self.memory_history) if self.memory_history else 0,
                'duration_minutes': self.time_stamps[-1] / 60 if self.time_stamps else 0
            }
        }
        
        with open('monitoring_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to monitoring_results.json")
        print(f"Average CPU Usage: {results['summary']['avg_cpu']:.1f}%")
        print(f"Peak CPU Usage: {results['summary']['max_cpu']:.1f}%")
        print(f"Average Memory Usage: {results['summary']['avg_memory']:.1f}%")
        
    def plot_results(self):
        """Create visualization of system usage"""
        if not self.time_stamps:
            print("No data to plot")
            return
            
        # Convert time to minutes
        time_minutes = [t/60 for t in self.time_stamps]
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # CPU usage plot
        ax1.plot(time_minutes, self.cpu_history, 'b-', linewidth=2)
        ax1.fill_between(time_minutes, self.cpu_history, alpha=0.3)
        ax1.set_ylabel('CPU Usage (%)', fontsize=12)
        ax1.set_ylim(0, 100)
        ax1.grid(True, alpha=0.3)
        ax1.set_title('System Resource Usage During HRM Training', fontsize=14, fontweight='bold')
        
        # Add average line
        avg_cpu = sum(self.cpu_history) / len(self.cpu_history)
        ax1.axhline(y=avg_cpu, color='r', linestyle='--', alpha=0.5, label=f'Average: {avg_cpu:.1f}%')
        ax1.legend()
        
        # Memory usage plot
        ax2.plot(time_minutes, self.memory_history, 'g-', linewidth=2)
        ax2.fill_between(time_minutes, self.memory_history, alpha=0.3)
        ax2.set_ylabel('Memory Usage (%)', fontsize=12)
        ax2.set_xlabel('Time (minutes)', fontsize=12)
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)
        
        # Add average line
        avg_mem = sum(self.memory_history) / len(self.memory_history)
        ax2.axhline(y=avg_mem, color='r', linestyle='--', alpha=0.5, label=f'Average: {avg_mem:.1f}%')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('system_usage.png', dpi=100, bbox_inches='tight')
        print(f"Plot saved to system_usage.png")
        
        # Show plot if possible
        try:
            plt.show()
        except:
            pass

def main():
    """Run system monitoring"""
    monitor = WindowsMonitor()
    
    # Check if training is running
    python_processes = [p for p in psutil.process_iter(['name']) if 'python' in p.info['name'].lower()]
    
    if len(python_processes) > 1:
        print(f"Found {len(python_processes)} Python processes running")
        print("Assuming training is active...\n")
    else:
        print("Warning: No other Python processes detected")
        print("Make sure to start training in another terminal\n")
    
    # Run monitoring (default 60 minutes, check every 5 seconds)
    monitor.run(duration_minutes=60, interval_seconds=5)

if __name__ == "__main__":
    main()