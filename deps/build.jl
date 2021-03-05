using KitBase.PyCall

try
    using Conda
    Conda.add_channel("conda-forge")
    Conda.add("tensorflow")
    tf = pyimport("tensorflow")
catch
    cmd = `pip3 install tensorflow --user`
    run(cmd)
    tf = pyimport("tensorflow")
end