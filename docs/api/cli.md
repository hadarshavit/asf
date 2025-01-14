# Commandline Interface

The CLI of ASF is a simple to allow you to run a training script of a selector asynchronously. It also contains a method to easily build this commandline call from existing variables, e.g. if you already have a Selector variable created in your script, you can simply call `asf.cli.cli_train.build_cli_command`, specifying the paths to the feature/performance data and where the created model should be written.
