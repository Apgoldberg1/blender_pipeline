import subprocess

def run_ns_train(folder_name, GPU_ID):
    command = f"conda activate ns && CUDA_VISIBLE_DEVICES={GPU_ID} ns-train nerfacto --vis viewer --machine.num-devices 1 --data outputs/{folder_name}"
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print("Error running ns-train:", e)

