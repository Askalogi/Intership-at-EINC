#! THESE ARE THE NOTES FOR THE HARDWARE IMPLEMETENTIONS !#


    #! Here we have the step by step guide given throug the link :
    #? https://github.com/electronicvisions/hxtorch 
    
       # 1) Most of the following steps will be executed within a singularity container
        #    To keep the steps clutter-free, we start by defining an alias
        shopt -s expand_aliases
        alias c="singularity exec --app dls /containers/stable/latest"

        # 2) Prepare a fresh workspace and change directory into it
        mkdir workspace && cd workspace

        # 3) Fetch a current copy of the symwaf2ic build tool
        git clone https://github.com/electronicvisions/waf -b symwaf2ic symwaf2ic

        # 4) Build symwaf2ic
        c make -C symwaf2ic
        ln -s symwaf2ic/waf

        # 5) Setup your workspace and clone all dependencies (--clone-depth=1 to skip history)
        c ./waf setup --repo-db-url=https://github.com/electronicvisions/projects --project=hxtorch

        # 6) Load PPU cross-compiler toolchain (or build https://github.com/electronicvisions/oppulance)
        module load ppu-toolchain

        # 7) Build the project
        #    Adjust -j1 to your own needs, beware that high parallelism will increase memory consumption!
        c ./waf configure
        c ./waf build -j1

        # 8) Install the project to ./bin and ./lib
        c ./waf install

        # 9) If you run programs outside waf, you'll need to add ./lib and ./bin to your path specifications
        export SINGULARITYENV_PREPEND_PATH=`pwd`/bin:$SINGULARITYENV_PREPEND_PATH
        export SINGULARITYENV_LD_LIBRARY_PATH=`pwd`/lib:$SINGULARITYENV_LD_LIBRARY_PATH
        export PYTHONPATH=`pwd`/lib:$PYTHONPATH

