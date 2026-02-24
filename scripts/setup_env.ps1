$ErrorActionPreference = "Stop"

$envName = "aesthetic312"
$pythonVersion = "3.12"
$pkgDir = "D:\conda_pkgs"
$envsDir = "D:\conda_envs"
$envPath = Join-Path $envsDir $envName

New-Item -ItemType Directory -Path $pkgDir -Force | Out-Null
New-Item -ItemType Directory -Path $envsDir -Force | Out-Null

$env:CONDA_PKGS_DIRS = $pkgDir
$env:CONDA_ENVS_PATH = $envsDir

if (Test-Path $envPath) {
    conda env remove -p $envPath -y
}

conda create -p $envPath python=$pythonVersion -y

conda run -p $envPath python -m pip install --upgrade pip
conda run -p $envPath pip install --no-cache-dir -r requirements.txt
conda run -p $envPath pip uninstall -y torch torchvision torchaudio

conda install -p $envPath pytorch=2.5.1 torchvision=0.20.1 torchaudio=2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y --force-reinstall

$env:KMP_DUPLICATE_LIB_OK = "TRUE"
conda run -p $envPath python -m torch.utils.collect_env
