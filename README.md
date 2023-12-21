# WeatherXM QoD

## Build
The following script creates a python environment using `poetry` and `pyenv` (need to be installed beforehand).
For specific dependency groups check the Makefile.
```bash
    brew install pyenv

    pyenv install 3.10.12

    pyenv local 3.10.12

    pip install poetry

    make install-all
```

The default venv installation path can be changed using:
```bash
    poetry config virtualenvs.path
```

## Local Docker run

```bash
    docker build -t wxm-qod:local .

    docker run \
      -v /datasets:/datasets \
      wxm-qod:local obc_sqc.iface.file_model_inference\
      --device_id <device_id> \
      --date <date> \
      --day1 <yesterday> \
      --day2 <today>
```

## IDE Settings

### Intellij import fix
Simply navigate to Project Settings -> Modules and then set 'Source Root' the top level 'src' folder.

### Plugins
* ruff (Intellij)

### Visualizations for profiling

cProfiler

    brew install graphviz

    gprof2dot -f pstats cprofile.pstats | dot -Tpng -o output.png
