ml purge

module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1 matplotlib/3.7.2-gfbf-2023a SciPy-bundle/2023.07-gfbf-2023a scikit-learn/1.3.1-gfbf-2023a

module load JupyterLab/4.0.5-GCCcore-12.3.0

source /folder1/folder2/venv/bin/activate
python -m ipykernel install --user --name=venv_sd_vae_cp --display-name="Virtual env."

jupyter lab --config="${CONFIG_FILE}"