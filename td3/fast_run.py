import xuance
import multiprocessing

#设置多进程启动方式为 'spawn'，避免与父进程冲突
multiprocessing.set_start_method('spawn', force=True)
# 使用简单的环境或调试模式，确认环境是否正常运行
runner = xuance.get_runner(method='maddpg',
                           env='mpe',
                           env_id='simple_spread_v3',
                           is_test=False)

# 确保启动前环境已就绪
try:
    runner.run()
except Exception as e:
    print(f"Error occurred during runner execution: {e}")
