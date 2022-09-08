# Isobar Sampler

## 1. Description

Allows sampling isobars nuclear configurations for running initial conditions for hydrodynamic simulations of ultrarelativistic nucleus-nucleus collisions in an efficient manner. 

It works by creating a bank of 'seeds', random numbers that can be used to calculate nucleon positions within different nuclear structures. These seeds are then used to generate nuclear configurations, consisting of an array of shape `(number_configurations, number_nucleons, 3)`, which stores nucleon positions `{x, y, z}`, in Cartesian coordinates.

Results are saved in HDF format and can be used with T<sub>R</sub>ENTo.

## 2. Usage

The code works in two stages. 

### 2.1. Create Seeds

First, a bank of 'seeds' must be created, by running

`./exec/make_seeds.py [seed_conf_file] `

The output is saved in HDF format, in a database named `isobar_seeds`.

##### Example configuration file:
An example configuration file can be found in `examples/seeds-conf.yaml`:
```
isobar_seeds:
   description: Configurations for making list of seeds for nucleon positions.
   
   number_nucleons: 
      description: Mass number A of the nuclei.
      value: 208
   
   number_configs:
      description: How many sets of nucleon positions to sample?
      value: 10000
         
   output_file:
      description: Path where to save list of seeds for nucleon positions.
      filename: nucleon-seeds.hdf
```


### 2.2. Build Nuclear Configurations

Nuclear configurations are built by running

`./exec/build_isobars.py [isobar_conf_file]`

##### Example configuration file:
An example configuration file can be found in `examples/isobars-conf.yaml`:
```
isobar_samples:
   description: Options for the isobar nucleon-position samples
   number_configs:
     description: Number of configurations to be sampled.
     value: 10000
   number_nucleons: 
      description: Mass number A of the nuclei.
      value: 208    
   seeds_file:
     description: Input file with list of seeds for nucleon positions.
     filename: nucleon-seeds.hdf
   output_path:
      description: Output directory where to save 
      dirname: test

isobar_properties:
   description: Nuclear properties of isobars to be sampled. Entries = isobar1, isobar2, ... Results are saved to isobar_name.hdf
   
   isobar1:
     isobar_name: WS1
     step_radius:
       description: Step-function radius in diffuse step function parametrization.
       value: 5.20678
     diffusiveness:
       description: Gaussian blurring width in diffuse step function parametrization.
       value: 0.772445
     beta_2:
       description: Quadrupolar deformation beta_2 of isobar.
       value: 0
     beta_3:
       description: Octupolar deformation beta_3 of isobar.
       value: 0
(...)
```

The output is saved in separate HDF files in the specified output path, with the same name as each of the isobars, in databases also having the same name. These databases contain an array of shape `(number_configurations, number_nucleons, 3)`, which stores nucleon positions `{x, y, z}`, in Cartesian coordinates.

## 3. Usage with T<sub>R</sub>ENTo

These nuclear configurations can be used within T<sub>R</sub>ENTo to generate initial conditions. The syntax is as follows:

`trento isobar.hdf isobar.hdf --random-seed random_seed [...]`

where the random seed should be specified for the different isobars so T<sub>R</sub>ENTo picks corresponding configurations and applies the same random rotation.


