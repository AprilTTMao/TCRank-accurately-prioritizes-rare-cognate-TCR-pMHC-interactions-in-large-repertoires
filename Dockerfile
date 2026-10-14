FROM condaforge/miniforge3:25.3.1-0@sha256:1e65803d646bf8503728001dddb5756f3e8235e0883c19b5a5ad1da85cff0905
USER root
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /opt/conda/bin/conda /usr/local/bin/conda
RUN conda create -y -n TCR-PEP_Platform --override-channels -c conda-forge -c bioconda \
      python=3.10.19 pip=25.3 anarci=2024.05.21 hmmer=3.3.2 \
    && conda create -y -n TCR-PEP_Platform_oldversion --override-channels -c conda-forge python=3.10.19 pip=25.3 \
    && conda clean -afy
COPY deploy/requirements-runtime.txt deploy/requirements-legacy.txt /tmp/
COPY deploy/constraints-runtime.txt deploy/constraints-legacy.txt /tmp/
RUN conda run -n TCR-PEP_Platform python -m pip install --no-cache-dir \
      torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu \
    && conda run -n TCR-PEP_Platform python -m pip install --no-cache-dir -r /tmp/requirements-runtime.txt -c /tmp/constraints-runtime.txt \
    && conda run -n TCR-PEP_Platform_oldversion python -m pip install --no-cache-dir -r /tmp/requirements-legacy.txt -c /tmp/constraints-legacy.txt \
    && conda run -n TCR-PEP_Platform python -m pip check \
    && conda run -n TCR-PEP_Platform_oldversion python -m pip check
ENV PATH=/opt/conda/bin:/opt/conda/envs/TCR-PEP_Platform/bin:$PATH \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" \
    HOME=/tmp MPLCONFIGDIR=/tmp/matplotlib HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true \
    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 TF_CPP_MIN_LOG_LEVEL=2 TCRANK_IN_CONTAINER=1
COPY deploy/check_runtime.py /app/deploy/check_runtime.py
RUN /opt/conda/envs/TCR-PEP_Platform/bin/python /app/deploy/check_runtime.py --imports-only
COPY build-assets/Predict_methods/ /opt/tcrank/Predict_methods/
COPY build-assets/ranker_artifacts/ /app/core/ranker_artifacts/
COPY build-assets/HLA-34position.csv /app/core/HLA-34position.csv
COPY core/ /app/core/
COPY TCR_run.py /app/
WORKDIR /app/core
USER 10001:10001
ENTRYPOINT ["/opt/conda/envs/TCR-PEP_Platform/bin/python", "/app/TCR_run.py"]
