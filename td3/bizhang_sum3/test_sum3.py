import subprocess

def run_script(script_name):
    subprocess.Popen(['python3', script_name])

if __name__ == '__main__':
    scripts = ['test_velodyne_td3_car1.py', 'test_velodyne_td3_car2.py', 'test_velodyne_td3_car3.py']

    processes = []
    for script in scripts:
        process = run_script(script)
        processes.append(process)

    # 可选：等待所有进程完成
    for process in processes:
        process.wait()
