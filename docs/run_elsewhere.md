# Running Elsewhere (Oracle Cloud Free Tier, VPS, or Local Docker)

Because `main.py` is 100% runner-agnostic (reading only CLI arguments and environment variables), you can move off GitHub Actions at any time without changing a single line of code.

## Docker Quickstart

```bash
docker build -f docker/Dockerfile -t recap-pipeline .

docker run --rm \
  -v "$(pwd)/projects:/app/projects" \
  -v "/tmp/recap_work:/tmp/recap_work" \
  -e GEMINI_API_KEY="$GEMINI_API_KEY" \
  recap-pipeline run \
  --config config.yaml \
  --slug my_film \
  --video-source "https://archive.org/details/..." \
  --mode full_auto \
  --publish-target artifact_only
```

## Oracle Cloud / Linux VPS

1. Install Python 3.11, `ffmpeg`, `fonts-dejavu-core`, and `libass9`.
2. `pip install -r requirements.txt`.
3. Export your secrets as environment variables and invoke `python main.py run ...`.
4. Increase `runtime.threads` and `encode.parallel_shot_jobs` in `config.yaml` if your VM has more than 2 vCPUs.
