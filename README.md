https://docs.sunfounder.com/projects/picar-x-v20/en/latest/

raspberry pi connect: https://www.raspberrypi.com/software/connect/
--> Desktop via browser into pi

1. Install all modules https://docs.sunfounder.com/projects/picar-x-v20/en/latest/python/python_start/install_all_modules.html (currently on global python environment only to be able to use the robohat)
2. Install "sudo pip3 install -r requirements.txt --break-system-packages" (toDo: need to ecapsulate step 1 and 2 to docker because sudo installation is not a good style)
3. Install uv: https://docs.astral.sh/uv/getting-started/installation/
4. Set up venv with "uv venv .venv --python 3.12"
5. activate venv: "source .venv/bin/activate"
6. sync: "uv sync"
