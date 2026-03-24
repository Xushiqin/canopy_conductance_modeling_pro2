# Model comparison pipeline with three SWC scenarios

## Simulation scenarios
This package evaluates each model under three scenarios:

1. `noSWC`
2. `withSWC_layer_1`
3. `withSWC_profile_mean`

The soil moisture parameter `n` is fitted separately for:
- `SWC_layer_1`  -> `n_layer_1`
- `SWC_profile_mean` -> `n_profile_mean`

For sites with only one SWC layer, `SWC_profile_mean` is expected to be `NaN`,
and the `withSWC_profile_mean` scenario is skipped automatically.

## Models included
- Jarvis
- BBL (`g0 = 0`, fit only `g1`)
- Medlyn (`g0 = 0`, fit only `g1`)
- RF_Rs_TA_VPD_leaf
- RF_GPP_VPD_leaf
- RF_GPP_VPD_leaf_TA_Rs

## RF settings
All RF models use:
- `n_estimators = 300`
- `max_depth = None`
- `min_samples_leaf = 2`
- `random_state = 42`
- `n_jobs = -1`

## Expected input structure
The script reads preprocessed site files from:
- `./fluxnet_data_model_comparison/DD/*.csv`
- `./fluxnet_data_model_comparison/HH/*.csv`

Each input file should contain at least:
- `TIMESTAMP`
- `gs_m_s-1`
- `VPD_leaf`
- `GPP`
- `Rs`
- `TA`
- `SWC_layer_1`
- `SWC_profile_mean`
- `SWC_layer_count`

The loader first tries `skiprows=[1]` to handle a units row on line 2.

## Output structure
Running:

    python run_model_comparison.py

creates:

./model_output/
├── DD/
│   ├── predictions/
│   └── summary/
└── HH/
    ├── predictions/
    └── summary/

Each site has:
- one predictions file
- one summary file

Each time scale also has:
- `DD_combined_summary.csv`
- `HH_combined_summary.csv`

## Required packages
Install with:

    conda install numpy pandas scipy scikit-learn

or

    pip install numpy pandas scipy scikit-learn
