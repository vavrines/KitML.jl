using Conda
#Conda.add_channel("conda-forge")
#Conda.add("tensorflow")
Conda.pip_interop(true)
Conda.pip("install", "tensorflow")

cmd = `pip3 install tensorflow --user`
run(cmd)
