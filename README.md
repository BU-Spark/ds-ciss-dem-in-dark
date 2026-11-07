
<h1 align="center">
  <br>
  <a href="https://www.bu.edu/spark/" target="_blank"><img src="https://www.bu.edu/spark/files/2023/08/logo.png" alt="BUSpark" width="200"></a>
  <br>
  Democracy in the Dark
  <br>
</h1>

<h4 align="center">Mapping electricity access and political distribution across Ghana's constituencies.</h4>

<p align="center">
  <a href="#key-features">Key Features</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#project-description">Project Description</a> •
  <a href="#data-locations">Data Locations</a>
</p>

## Key Features

This repo builds a pipeline that links nighttime light data to Ghana's constituencies and election results.

* `scripts/nightlights/`: pulls VIIRS nighttime light data, builds the analytic panel, checks data quality, and draws the main maps.
  - Includes a `GEE/` subfolder with the Earth Engine scripts that pull the raw data.
* `scripts/seasonal_pattern/`: groups constituencies into climate zones and measures their seasonal light patterns.
* `scripts/election_impact/`: tests whether competitive elections line up with nightlight changes around each vote.

See `scripts/SCRIPTS_GUIDE.md` for the run order and the inputs and outputs of each script.

Results land in:
* `result/nightlights/`: panel outputs and maps from the nightlights scripts.
* `result/seasonal_zones/`: seasonal decomposition outputs by climate zone.
* `result/election_impact/`: election versus nightlights comparison outputs.

## How To Use

To clone and run this application, you'll need <a href="https://git-scm.com" target="_blank">Git</a>
From your command line:

```bash
# Clone this repository
$ git clone https://github.com/BU-Spark/ds-ciss-dem-in-dark.git

# Further Instructions
...
```

Create a new branch from dev, add changes on the new branch you just created.

You will want to look into <a href="https://git-scm.com/docs/git-branch" target="_blank">git branch</a> and <a href="https://git-scm.com/docs/git-checkout" target="_blank">git checkout</a>

```bash
# Create and Checkout a new branch if it doesn't exist
$ git checkout -b your-branch main
...
```

Open a Pull Request to dev. Add your PM and TPM as reviewers. 

At the end of the semester during project wrap up open a final Pull Request to main from dev branch.
 
## Project Description

* This project studies whether electricity access across Ghana's constituencies tracks political outcomes.
* It looks at incumbency, electoral competitiveness, and changes in access around election periods.
* It also accounts for seasonal swings in access driven by the dry season, roughly November through March.

## Data locations

See <a href="dataset-documentation/DATASETDOC.md">dataset-documentation/DATASETDOC.md</a> for full details on every dataset used in this project.
