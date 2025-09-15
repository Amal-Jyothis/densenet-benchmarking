# Default values
OUTPUT_DIR="./results"
GPU_ENABLED="false"

while [[ $# -gt 0 ]]; do
  case $1 in
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --gpu-enabled)
      GPU_ENABLED="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "Building Docker image..."
OUTPUT_DIR=$OUTPUT_DIR GPU_ENABLED=$GPU_ENABLED docker-compose build

echo "Running container..."
OUTPUT_DIR=$OUTPUT_DIR GPU_ENABLED=$GPU_ENABLED docker-compose up --abort-on-container-exit --remove-orphans

echo "Cleaning up..."
docker-compose down --volumes --remove-orphans

echo "Done"