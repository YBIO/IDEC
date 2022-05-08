import os
import sys
import time
 
cmd = 'nohup sh run.sh > logs/train_ADE_100-50_v3.log 2>&1 &'
 
 
def gpu_info():
    gpu_status = os.popen('nvidia-smi | grep %').read().split('|')
    gpu_memory0 = int(gpu_status[2].split('/')[0].split('M')[0].strip())
    gpu_power0 = int(gpu_status[1].split('   ')[-1].split('/')[0].split('W')[0].strip())
    gpu_memory1 = int(gpu_status[6].split('/')[0].split('M')[0].strip())
    gpu_power1 = int(gpu_status[5].split('   ')[-1].split('/')[0].split('W')[0].strip())
    gpu_memory2 = int(gpu_status[10].split('/')[0].split('M')[0].strip())
    gpu_power2 = int(gpu_status[9].split('   ')[-1].split('/')[0].split('W')[0].strip())
    gpu_memory3 = int(gpu_status[14].split('/')[0].split('M')[0].strip())
    gpu_power3 = int(gpu_status[13].split('   ')[-1].split('/')[0].split('W')[0].strip())
     # four GPUs
    gpu_power = [gpu_power0, gpu_power1, gpu_power2, gpu_power3]
    gpu_memory = [gpu_memory0, gpu_memory1, gpu_memory2, gpu_memory3]

    return gpu_power, gpu_memory
 
 
def narrow_setup(interval=30): # refresh every intervval seconds
    gpu_power, gpu_memory = gpu_info()
    i = 0
    while gpu_memory[0] > 100 and gpu_memory[1] > 100 and gpu_memory[2] > 100 and gpu_memory[3] > 100:  # set waiting condition
        gpu_power, gpu_memory = gpu_info()
        i = i % 5
        symbol = 'monitoring: ' + '>' * i #+ ' ' * (10 - i - 1) + '|'
        gpu_power_str = 'gpu power:%d, %d, %d, %dW |' % (gpu_power[0], gpu_power[1], gpu_power[2], gpu_power[3])
        gpu_memory_str = 'gpu memory:%d, %d, %d, %d MiB |' % (gpu_memory[0], gpu_memory[1], gpu_memory[2], gpu_memory[3])
        sys.stdout.write('\r' + gpu_memory_str + ' ' + gpu_power_str + ' ' + symbol)
        sys.stdout.flush()
        time.sleep(interval)
        i += 1
    print('\n ###Executing===> \n' + cmd)
    os.system(cmd)
 
 
if __name__ == '__main__':
    narrow_setup()