# ATSim 2.0

Public executable files and test cases for ATSim 2.0.

ATSim 2.0 is associated with papers published at ISEDA 2025 and ASICON 2025.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Input Files](#input-files)
- [Output Files](#output-files)
- [Command-Line Usage](#command-line-usage)
- [Configuration Reference](#configuration-reference)
- [Features](#features)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Contact](#contact)

## Overview

ATSim is a thermal simulation engine for three-dimensional integrated circuits
developed by Peking University. The tool targets advanced packaging and
high-performance computing systems, and provides chip-package co-analysis with
high accuracy and high computational efficiency.

The first-generation algorithm used a coarse-fine two-level grid strategy: a
global coarse grid was solved first, followed by local refinement around hotspot
regions. ATSim 2.0 introduces multiscale adaptive meshing, enabling hierarchical
modeling from transistor-level chip structures to package-level heat spreaders
and heatsinks while preserving physical fidelity and improving runtime.

### Highlights

- **Multiscale thermal solving**: adaptive meshing balances simulation speed and
  accuracy across heterogeneous package structures.
- **Steady-state and transient analysis**: supports constant-power simulations
  and time-varying thermal behavior.
- **Advanced packaging support**: models 2D-IC, 2.5D-IC, MCM, 3D-IC, 3.5D-IC,
  and related integration styles.
- **Flexible power modeling**: supports constant, lookup-table, linear, and
  exponential power models, with optional temperature feedback.
- **Modular input structure**: XML-based package descriptions organize geometry,
  materials, power, floorplans, and reusable simulation assets.

## Quick Start

### Recommended Platform

ATSim 2.0 is recommended for Ubuntu 20.04.

### Basic Workflow

1. Prepare the package XML file, simulation configuration, material library,
   floorplan CSV files, power model YAML file, and power trace CSV files.
2. Run the ATSim executable:

   ```bash
   ./bin/ATSim -xml cases/3DIC/3DIC.xml -config cases/3DIC/steady.config --output_path ./rst --plot_flag 1
   ```

3. Check the output directory for raw temperature data and optional plots.

## Input Files

ATSim simulations are driven by five major input file types.

### 1. Package Geometry XML

The XML file is the primary input file. It describes the full package stack from
chip active layers to external thermal structures.

- `<Package>` is the root node and defines the package name and unit. The unit
  must be meters.
- `<MaterialLib>` and `<PowerLib>` reference the material configuration file and
  power model library.
- `<Location>` defines package-level translation and rotation. The current
  version supports `R0`.
- `<Component>` represents a physical entity such as a chip, TIM, heatsink, or
  passive block.
- `<Layer>` defines material layers inside a component.
- `<Chiplet>` describes nested chiplets inside an active component.

For active components (`type="Active"`), each functional layer may reference its
own floorplan and power files through `<Floorplan>` and `<Power>` tags. This
allows heterogeneous 2.5D and 3D package structures to be described in a single
XML hierarchy.

`<Geometry>` defines component length and width, while layer `<Geometry>` defines
thickness. `<XYOrigin>` defines the component origin. For active components, the
origin should be `XYOrigin`. For other components, the origin can be an absolute
coordinate `(X,Y)` or `Center`, which aligns the component with the active
component center.

Filling materials can be defined at the `<Component>`, `<Chiplet>`, `<Layer>`,
and floorplan levels. Inner definitions have higher priority and override outer
definitions.

### 2. Material Configuration

Material properties are defined in `.config` files. Each material block should
include key thermal parameters such as:

- `conductivity`
- `volumetric_capacity`

Material names must exactly match the `FillingMaterial` values used in the XML
file.

The current material format supports nonlinear materials and orthotropic
anisotropic materials.

### 3. Power Model YAML

The power model YAML file defines reusable power sources such as `source1`,
`source2`, and so on. Each source may contain three power components:

- `internal`: static internal power.
- `switch`: dynamic switching power.
- `leakage`: leakage power, which can be constant, linear, exponential, or a
  temperature lookup table such as `[45C, 2W]` and `[85C, 4W]`.

When `power_type` is set to `util` in the simulation configuration, the power CSV
defines the utilization of each source. Total power is evaluated as:

```text
internal + switch * utilization + leakage(temperature)
```

When `power_type` is set to `value`, or when no power model YAML is provided,
the power CSV directly defines the power value for each source.

### 4. Power Trace CSV

Power trace CSV files map floorplan macros to power sources or direct power
values.

- Each column corresponds to a macro or power source entry.
- Each row corresponds to one time step.
- For steady-state simulations, only the first row is used.
- Macro names referenced by the power CSV must exist in the corresponding
  floorplan CSV.
- Sources referenced by the power CSV must be defined in the YAML power library
  when a power library is used.
- Macros not listed in the power CSV are treated as having no heat source.

### 5. Floorplan CSV

Floorplan CSV files define the position, size, and optional material label of
each macro.

Expected columns include:

```text
UnitName, X, Y, Z, Length, Width, Thickness, Label
```

The fields represent the macro name, lower-left coordinate, dimensions, and
material label. `Thickness` and `Label` may be omitted when they are provided by
the enclosing XML/layer definition.

Macro names must be unique across different floorplan CSV files. TSV
distributions can also be defined in floorplan files; see the included examples
for details.

### 6. Simulation Configuration

Simulation `.config` files control mesh density, parallelism, solving mode, and
other runtime parameters. See the example cases for practical settings.

Note: transient simulation currently does not support nonlinear materials.

## Output Files

### Raw Temperature Field

Raw simulation results are written under the `raw` folder in the output path.
Each result contains seven columns:

```text
xmin, xmax, ymin, ymax, zmin, zmax, temperature
```

The first six columns define the grid-cell geometry. The final column is the
temperature at the center node of that grid cell. For transient simulations,
results are stored for each time step.

### Visualization Files

When plotting is enabled with `--plot_flag 1`, ATSim automatically generates
visualization files for power and temperature distributions on powered layers.
Steady-state simulations generate image files, while transient simulations can
generate animated GIF files.

More visualization capabilities will be provided in a future GUI version.

## Command-Line Usage

ATSim is launched from the command line:

```bash
./bin/ATSim -xml <package_definition.xml> -config <simulation_config.config>
```

Common options:

- `--output_path <path>`: output directory. The default is the current working
  directory.
- `--plot_flag 1`: enable visualization output.

Material `.config` files and power model `.yml` files are referenced from the XML
file and do not need to be passed as separate command-line arguments.

Example:

```bash
./bin/ATSim -xml cases/3DIC/3DIC.xml -config cases/3DIC/steady.config --output_path ./rst --plot_flag 1
```

## Configuration Reference

### XML Package Definition

The package XML uses a hierarchical structure to describe geometry, materials,
floorplans, and power bindings.

Important rules:

- The root `<Package>` unit must be meters.
- `<MaterialLib>` and `<PowerLib>` paths are resolved from the simulation working
  directory unless absolute paths are used.
- Each `<Component>` can contain multiple `<Layer>` elements.
- Active components can contain nested `<Chiplet>` components.
- A layer that contains functional macros should define both `<Floorplan>` and
  `<Power>` tags.
- Material definitions closer to the macro level override outer definitions.

### Material Configuration

Material configuration files support:

- temperature-dependent nonlinear materials
- orthotropic anisotropic thermal conductivity
- reusable material blocks referenced by XML names

### Power and Floorplan Binding

Floorplan CSV files define where macros are located. Power CSV files define
which macros dissipate power and how much power they dissipate. ATSim matches
the two files by macro name.

If a macro exists in the floorplan but not in the power CSV, it is treated as an
unpowered region.

## Features

### Flexible Geometry Definition

ATSim supports 2D, 2.5D, 3D, and 3.5D package structures through a flexible XML
description.

### Flexible Material Definition

The material system supports nonlinear and anisotropic materials.

### Flexible Power Models

Power models can be specified independently for each heat source using constant,
linear, exponential, or lookup-table forms.

### Multiscale Adaptive Meshing

Adaptive meshing refines regions with large thermal gradients or abrupt material
changes, improving accuracy while controlling runtime.

### TSV Modeling

TSV structures can be modeled to capture their impact on heat conduction paths.

### Transient Thermal Analysis

Time-domain simulation supports time-varying power traces and can be used to
study dynamic temperature response.

## Examples

### 1. Multilayer 3D-IC

The `cases/3DIC` example contains two active layers, such as logic and cache,
connected through hybrid bonding. Each active layer uses its own floorplan and
power files. This case can be used to test nonlinear materials, nonlinear power
models, steady-state simulation, and transient simulation.

### 2. GPU-HBM Package

The `cases/GPUHBM` example is a simplified model based on the NVIDIA V100
architecture. It contains one compute die and multiple HBM stacks. The compute
die and HBM stacks have different thickness distributions, forming a 3.5D
package structure. The XML file defines the multi-chip layout, and separate
floorplan files assign power to different dies.

## Troubleshooting

### File Not Found or Path Errors

Check all file paths, including floorplan, power, material, and power library
paths. Relative paths should be valid relative to the working directory where
ATSim is launched.

### Material Name Mismatch

Every `FillingMaterial="..."` value used in the XML file must have a matching
material block in the `.config` material file.

### Out-of-Memory Errors or Stalled Solves

Reduce mesh density or adaptive refinement depth. For large models, use
multi-core parallel execution when possible.

### Unexpected Temperature Results

If the temperature field remains near ambient temperature, check that:

- power CSV macro names match floorplan macro names
- referenced YAML power sources exist
- power values are nonzero
- the intended power file is referenced by the XML file

## Contact

For questions, issues, or support, please contact the developers:

- txzhu@pku.edu.cn
- qpwang@pku.edu.cn
