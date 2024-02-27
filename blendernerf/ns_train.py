import subprocess

def run_ns_train(folder_name, GPU_ID):
    command = f"conda deactivate && conda init zsh && conda activate ns && CUDA_VISIBLE_DEVICES={GPU_ID} ns-train splatfacto --vis viewer --machine.num-devices 1 --data {folder_name}"
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print("Error running ns-train:", e)

#TODO: write javascript creation code to be compatible with nerfstudio poses
def run_ns_render(folder_name, GPU_ID):
    for i, pose in enumerate(poses):
        # Construct command for ns-render
        command = [
            "ns-render",
            "--scene", scene_file,
            "--camera_position", ",".join(map(str, pose["position"])),
            "--camera_orientation", ",".join(map(str, pose["orientation"])),
            "--output", f"output_{i}.png"  # Output file name
        ]

        # Execute ns-render
        subprocess.run(command)

    print("Rendering complete.")


if __name__ == "__main__":
    run_ns_train("outputs/flowerpost", 6)
