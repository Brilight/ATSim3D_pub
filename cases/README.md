# ATSim 2.0 Example Cases

This directory contains runnable example cases for ATSim 2.0. The examples
demonstrate steady-state and transient thermal simulations for advanced
packaging structures.

## Directory Layout

```text
cases/
├── 3DIC/       # Two-active-layer 3D-IC example
└── GPUHBM/     # GPU-HBM package example with a TSV interposer
```

## Before Running

Start from the repository root and enter the `cases` directory:

```bash
cd ./cases
```

The commands below assume that `bin/ATSim` is available at the repository root.

## Example 1: 3D-IC

The `3DIC` case models a two-layer 3D integrated circuit with two powered active
layers. It can be used to test nonlinear steady-state simulation and nonlinear
transient simulation.

Enter the case directory:

```bash
cd ./3DIC
```

### Nonlinear Steady-State Simulation

```bash
../../bin/ATSim -xml ./3DIC.xml -config steady.config --output_path ./steady --plot_flag 1
```

This command creates a `steady` output directory. The `steady/raw` directory
contains the raw simulation result, including `steadytemp.npy`. When plotting is
enabled, ATSim also generates power and temperature distribution figures.

### Nonlinear Transient Simulation

```bash
../../bin/ATSim -xml ./3DIC.xml -config transient.config --output_path ./transient --plot_flag 1
```

This command creates a `transient` output directory. The `transient/raw`
directory stores per-step simulation results, including `res.npy`. When plotting
is enabled, ATSim also generates animated power and temperature visualizations.

Return to the `cases` directory before running the next example:

```bash
cd ..
```

## Example 2: GPU-HBM

The `GPUHBM` case models a GPU-HBM system. The interposer includes a `41 x 41`
TSV array.

Enter the case directory:

```bash
cd ./GPUHBM
```

### Nonlinear Steady-State Simulation

```bash
../../bin/ATSim -xml ./package.xml -config sim.config --output_path ./steady --plot_flag 1
```

This command creates a `steady` output directory. The `steady/raw` directory
contains the raw simulation result, including `steadytemp.npy`. When plotting is
enabled, ATSim also generates power and temperature distribution figures.

## Output Summary

Each example writes results to the directory specified by `--output_path`.

- `raw/`: raw numerical temperature results in NumPy format.
- image files: generated power and temperature distribution plots when
  `--plot_flag 1` is enabled.
- GIF files: generated for transient simulations when visualization is enabled.

## Notes

- Run commands from the corresponding case directory unless otherwise stated.
- Use Linux-style paths on Ubuntu, for example `../../bin/ATSim`.
- If an executable permission error occurs, run:

  ```bash
  chmod +x ../../bin/ATSim
  ```
