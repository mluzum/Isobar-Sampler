# Isobar Sampler

The official version of this code is available at <https://github.com/mluzum/Isobar-Sampler>.

## 1. Description

Allows sampling isobaric nuclear configurations for running initial conditions for hydrodynamic simulations of ultrarelativistic nucleus-nucleus collisions in an efficient manner. 

The method is described in the article [Eur.Phys.J.A 59 (2023) 5, 110](https://link.springer.com/article/10.1140/epja/s10050-023-01021-8) ([arXiv:2302.14026](https://arxiv.org/abs/2302.14026)), which should be cited with any results obtained from this code.  For the realistic correlation function, see the Quark Matter 2023 proceedings ([arXiv:2312.10129](https://arxiv.org/abs/2312.10129)), which should be cited alongside the original source for the nuclei from which the correlation function was extracted [Phys.Rev.C 101 (2020) 6, 061901](https://doi.org/10.1103/PhysRevC.101.061901) ([arXiv:1908.10231](https://arxiv.org/abs/1908.10231)).

It works by creating a bank of 'seeds', random numbers that can be used to calculate nucleon positions within different nuclear structures. These seeds are then used to generate nuclear configurations, consisting of an array of shape `(number_configs, number_nucleons, 3)`, which stores nucleon positions `{x, y, z}`, in Cartesian coordinates.  

Nuclei with arbitrary Woods-Saxon parameters (R,a), angular deformation parameters (beta_2, gamma, beta_3), and short-range correlations (currently either a step function with depth correlation_strangth and distance correlation_length or a realistic correlation function that can be scaled with a length_scale and/or strength_scale), can be prepared and efficiently compared to each other.  As long as the same seeds are used, the relative difference between nuclei can be accurately studied with few statistics.

Results are saved in HDF format and can be easily used, e.g., with T<sub>R</sub>ENTo.  (Note that T<sub>R</sub>ENTo must be modified to ensure synchronization of fluctuations beyond the nuclear configuration -- impact parameter, target/projectile orientation, gamma fluctautions, etc.  Such a modified version of T<sub>R</sub>ENTo will be released in the future.)

## 2. Usage

The code works in two stages. First a bank of 'seeds' is created and written to a file.  These seeds represent fluctuations in nucleon positions, and can be then  used to generate configurations with different nuclear shape and correlation properties ('isobars'), but which in a sense share the same respective fluctuations.   The same seed file can be used to generate additional compatible isobars in the future.

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
      value: 96
   
   number_configs:
      description: How many sets of nucleon positions to sample?
      value: 10000
         
   output_file:
      description: Path where to save list of seeds for nucleon positions.
      filename: nucleon-seeds.hdf

   number_of_parallel_processes:
       description: Number of processes to compute in parallel.  
       value: 40
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
      value: 96    
   seeds_file:
     description: Input file with list of seeds for nucleon positions.
     filename: nucleon-seeds.hdf
   output_path:
      description: Output directory where to save 
      dirname: test
   number_of_parallel_processes:
       description: Number of processes to compute in parallel. A value of -1 automatically selects the number of CPUs present.  See joblib.Parallel class.
       value: -1

isobar_properties:
   description: Nuclear properties of isobars to be sampled. Entries = isobar1, isobar2, ... Results are saved to isobar_name.hdf
   
   isobar1:
     isobar_name: WS1
     WS_radius:
       description: Woods-Saxon radius parameter R
       value: 5.09
     WS_diffusiveness:
       description: Woods-Saxon diffusiveness parameter a
       value: 0.46
     beta_2:
       description: Quadrupolar deformation beta_2 of isobar.
       value: 0.16
     gamma:
       description: Quadrupolar deformation angle (in radians).
       value: 0.5236
     beta_3:
       description: Octupolar deformation beta_3 of isobar.
       value: 0
     correlation_length:
       description: Radius of step-function correlation function C(r) in fm.
       value: 0.4
     correlation_strength:
       description: Depth of correlation function for r < correlation_length.  Should be >= -1.
       value: -1

   isobar2:
     isobar_name: WS2
(...)
```

The output is saved in separate HDF files in the specified output path, with the same name as each of the isobars, in databases also having the same name. These databases contain an array of shape `(number_configs, number_nucleons, 3)`, which stores nucleon positions `{x, y, z}`, in Cartesian coordinates.

## 3. Usage with T<sub>R</sub>ENTo

These nuclear configurations can be used within T<sub>R</sub>ENTo to generate initial conditions. However, a  modified version of Trento (to be released in the future) is necessary to properly synchronize fluctuations across corresponding nuclei.

The syntax is as follows:

`trento isobar.hdf isobar.hdf --random-seed random_seed --b-max b_max [...]`

where the random seed should be specified for the different isobars so T<sub>R</sub>ENTo picks corresponding configurations and applies the same random elements (impact parameter, rotations, gamma fluctuations, etc.). The maximum impact parameter must be fixed by hand as well, otherwise it will vary with the nuclei.  



